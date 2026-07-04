"""Chronological sequencing + prerequisite hints (spec §7C) + severed-pair linking (P4b).

Clips are presented in temporal order (earlier-in-video first). The dependency graph is
used only to attach optional "watch first" hints — never to reorder.

P4b (Wave 2 §16): two halves of ONE worked example that shipped as separate clips (the
audited kinematics bus problem — setup/givens in one clip, result 42s later in another,
``prerequisite_clips=[]``) are detected after sequencing. The pair is ALWAYS linked
(earlier sequence index into the later clip's ``prerequisite_clips`` + a
'continues clip N' note); a merge is ADDITIONALLY attempted only when the combined span
fits the ship cap, via a caller-supplied ``merge_fn`` that must re-judge the union and
return it ONLY on a clean verdict — otherwise the linked pair stands. Nothing is ever
killed here.

W25-F relaxation (the linker fired 0 times everywhere by construction): opener roles gain
practice_prompt + setup; the LINK scan covers non-adjacent later clips within
SEVERED_MAX_GAP_S (merges stay adjacency-gated); and the later-side test is pair-scoped —
an opener unit inside the later clip only blocks when answers edges / shared concepts tie
it to THIS pair's problem, not merely by existing (closure force-inlines required setup
into payoff clips, which made the old blanket test permanently false).
"""
from __future__ import annotations

from typing import Callable, Optional

# W25-F: practice_prompt (qP's most common opener — 27 units) and setup join the opener
# set; the linker previously fired 0 times partly because the commonest openers weren't
# openers at all.
SEVERED_OPENER_ROLES = frozenset({"example_setup", "problem_givens", "practice_prompt", "setup"})
SEVERED_PAYOFF_ROLES = frozenset({"result", "solution"})
SEVERED_MAX_GAP_S = 60.0


def attach_prerequisites(specs: list[dict], graph, units_by_id: dict) -> None:
    """Hint an earlier clip only for concepts this clip REQUIRES but does not itself
    introduce — a clip that contains its own definition needs no 'watch first' pointer."""
    def _units(s):
        return [units_by_id[u] for u in s.get("unit_ids", []) if u in units_by_id]

    for s in specs:
        mine = _units(s)
        intro = set().union(*[set(u.concepts_introduced) for u in mine]) if mine else set()
        req = set().union(*[set(u.concepts_required) for u in mine]) if mine else set()
        unmet = {c for c in (req - intro) if c}
        prereqs: list[int] = []
        if unmet:
            # precompute each earlier clip's introduced-set ONCE (not per concept)
            earlier = []
            for other in specs:
                if other is s or other["start"] >= s["start"]:
                    continue                                  # only EARLIER clips
                ou_units = _units(other)
                ou_intro = set().union(*[set(u.concepts_introduced) for u in ou_units]) \
                    if ou_units else set()
                earlier.append((other, ou_intro))
            for concept in unmet:
                for other, ou_intro in earlier:               # specs are in temporal order
                    if concept in ou_intro:
                        prereqs.append(other["sequence_index"])
                        # only the earliest for this concept — deliberate departure from
                        # hint-all-earlier-clips: fewer redundant "watch first" pointers;
                        # the earliest introducer suffices
                        break
        s["prerequisite_clips"] = sorted(set(prereqs))


def sequence_clips(specs: list[dict], graph, units_by_id: dict) -> list[dict]:
    specs.sort(key=lambda s: s["start"])
    for i, s in enumerate(specs):
        s["sequence_index"] = i + 1
    attach_prerequisites(specs, graph, units_by_id)
    return specs


# ── P4b: severed-pair detection + link/merge ─────────────────────────────────
def _spec_roles(s: dict, units_by_id: dict) -> set[str]:
    return {units_by_id[uid].role for uid in s.get("unit_ids", []) if uid in units_by_id}


def _spec_topics(s: dict, units_by_id: dict) -> set[str]:
    out: set[str] = set()
    for uid in s.get("unit_ids", []):
        u = units_by_id.get(uid)
        if u is not None:
            t = (u.node_id or u.topic or "").strip()
            if t:
                out.add(t)
    return out


def _same_topic(a: dict, b: dict, units_by_id: dict) -> bool:
    """Same-topic gate: clips are DIFFERENT-topic only when both sides carry topic
    evidence (unit node_id/topic) and none of it overlaps — with no evidence on a side,
    the role grammar + the 60s gap cap remain the deciding signals."""
    ta, tb = _spec_topics(a, units_by_id), _spec_topics(b, units_by_id)
    if not ta or not tb:
        return True
    return bool(ta & tb)


def _pair_opener_in_later(earlier: dict, later: dict, units_by_id: dict, graph=None) -> bool:
    """W25-F pair-scoped replacement for the blanket 'later has no opener roles' test —
    which closure defeated by construction (required problem statements are force-inlined
    into every payoff clip, so the linker fired 0 times everywhere). An opener-role unit
    inside the LATER clip blocks the pair ONLY when it demonstrably belongs to THIS pair's
    problem (i.e. the later clip restates its own setup and is genuinely self-contained):
    (a) answers edges — a payoff unit in the later clip directly answers that in-clip
    opener; (b) shared concepts — the opener trades concepts with the earlier clip's
    units. An opener with NEITHER tie is some OTHER problem's setup that drifted into the
    span (the next practice prompt, an unrelated example) and must not veto the link;
    linking kills nothing, so the relaxation is safe by construction."""
    later_ids = [uid for uid in later.get("unit_ids", []) if uid in units_by_id]
    openers = [uid for uid in later_ids if units_by_id[uid].role in SEVERED_OPENER_ROLES]
    if not openers:
        return False
    answered: set[str] = set()                # openers a later payoff unit directly answers
    if graph is not None:
        for uid in later_ids:
            if units_by_id[uid].role in SEVERED_PAYOFF_ROLES:
                for e in graph.needs(uid, ("answers",)):
                    answered.add(e.target)
    earlier_concepts: set[str] = set()
    for uid in earlier.get("unit_ids", []):
        u = units_by_id.get(uid)
        if u is not None:
            earlier_concepts.update(c for c in u.concepts_introduced if c)
            earlier_concepts.update(c for c in u.concepts_required if c)
    for uid in openers:
        if uid in answered:
            return True                       # the later payoff answers its OWN in-clip prompt
        u = units_by_id[uid]
        mine = {c for c in (*u.concepts_introduced, *u.concepts_required) if c}
        if mine & earlier_concepts:
            return True                       # this pair's setup, restated in the later clip
    return False


def is_severed_pair(earlier: dict, later: dict, units_by_id: dict,
                    max_gap_s: float = SEVERED_MAX_GAP_S, graph=None) -> bool:
    """One instructional event cut in two: the EARLIER clip has opener roles
    (example_setup|problem_givens|practice_prompt|setup) but no payoff (result|solution),
    the LATER clip has the payoff but no opener unit belonging to THIS pair's problem
    (W25-F pair-scoped check — answers edges / shared concepts, see _pair_opener_in_later),
    same topic, and the hole between them is at most ``max_gap_s``."""
    if later["start"] - earlier["end"] > max_gap_s:
        return False
    if not _same_topic(earlier, later, units_by_id):
        return False
    er = _spec_roles(earlier, units_by_id)
    lr = _spec_roles(later, units_by_id)
    return (bool(er & SEVERED_OPENER_ROLES) and not (er & SEVERED_PAYOFF_ROLES)
            and bool(lr & SEVERED_PAYOFF_ROLES)
            and not _pair_opener_in_later(earlier, later, units_by_id, graph))


def link_severed_pairs(specs: list[dict], graph, units_by_id: dict, max_dur_s: float,
                       merge_fn: Optional[Callable[[dict, dict], Optional[dict]]] = None
                       ) -> list[dict]:
    """P4b, run AFTER sequencing. Pass 1 attempts merges: a severed pair whose combined
    span fits ``max_dur_s`` is handed to ``merge_fn`` (which builds the union, re-judges
    its NEVER-judged text, and returns it only on a clean verdict); a successful merge
    replaces the pair and the list is re-sequenced (fresh indices + prerequisite hints).
    Merges stay ADJACENCY+cap gated (a union across an intervening clip would swallow it).
    Pass 2 ALWAYS links whatever remains severed — W25-F: the scan is no longer
    adjacent-only; each opener clip looks at EVERY later clip whose start is within
    SEVERED_MAX_GAP_S of its end (specs are start-sorted, so the scan breaks at the first
    gap overrun) and links the NEAREST matching payoff: the earlier clip's sequence index
    joins the later clip's prerequisite_clips (dedup'd) and a 'continues clip N' note is
    appended — the learner is told to watch part 1 first even when no merge is possible."""
    if merge_fn is not None:
        out: list[dict] = []
        merged_any = False
        i = 0
        while i < len(specs):
            s = specs[i]
            if i + 1 < len(specs):
                nxt = specs[i + 1]
                if (is_severed_pair(s, nxt, units_by_id, graph=graph)
                        and nxt["end"] - s["start"] <= max_dur_s):   # merge ONLY under the cap
                    merged = merge_fn(s, nxt)
                    if merged is not None:                           # clean re-judge → union ships
                        out.append(merged)
                        merged_any = True
                        i += 2
                        continue
            out.append(s)
            i += 1
        specs = out
        if merged_any:
            specs = sequence_clips(specs, graph, units_by_id)
    for i, earlier in enumerate(specs):
        for later in specs[i + 1:]:
            if later["start"] - earlier["end"] > SEVERED_MAX_GAP_S:
                break                          # start-sorted: every further clip is too far
            if not is_severed_pair(earlier, later, units_by_id, graph=graph):
                continue
            n = int(earlier.get("sequence_index", 0))
            later["prerequisite_clips"] = sorted(
                set(later.get("prerequisite_clips") or []) | {n})
            note = f"continues clip {n}"
            notes = list(later.get("notes") or [])
            if note not in notes:
                notes.append(note)
            later["notes"] = notes
            break                              # one pair per opener — the nearest payoff
    return specs
