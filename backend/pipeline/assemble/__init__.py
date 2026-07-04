"""Topic-dependent clip assembly (spec §6–9, §7B–C).

assemble_clips turns a cached Structure + a search topic into ordered, self-contained
clip specs: anchor selection → context closure → clip-only judge + repair → boundary
snapping (reusing the existing code) → quality filter → context cards → chronological
sequencing → severed-pair link/merge (P4b). Returns (clips_spec, notes, rejections); the
orchestrator applies the optional precise Whisper boundary pass and builds the final
embed/cut clip dicts.
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ... import config
from ..refine import _trim_start_after
from ..sentences import Sentence
from ..understand.models import Structure
from . import scoring
from .boundary_adapt import snap_candidates
from .candidates import (
    _anchor_eligible, build_arc_candidate, build_candidate, compute_anchor_budget,
    score_topic_relevance, select_anchors, select_anchors_planned, topic_matches_subject,
)
from .context_card import generate_context_card, generate_orientation_cards
from .contracts import choose_contract
from .graph import Graph
from .integrity import Rejection, merge_partb, true_contents
from .sequence import link_severed_pairs, sequence_clips
from .validate import (
    _card_clears, _card_rescuable, _card_rescue_verdict, _hard_core_ok,
    _verify_failure_reasons, confirm_kill, is_complete, judge_clip, judged_text_hash,
    validate_and_repair,
)

ProgressCb = Optional[Callable[[float, str], None]]


def _card_warning(s: dict) -> None:
    """A clip that NEEDS earlier context (referential prereqs or truncated closure) but has no
    card ships with an explicit, penalized marker instead of failing silently. A CARD5(b)
    suppressed card is deliberate (the first sentence already names the subject) — as when
    referential is empty, it carries no card and no missing-card penalty."""
    if s.get("card_suppressed"):
        return
    if not s.get("context_card") and (s.get("referential") or s.get("truncated")):
        s["warnings"] = tuple(set(s.get("warnings") or ()) | {"missing_context_card"})


def _first_sentence_names_subject(s: dict, sentences: list[Sentence], units_by_id, topic: str) -> bool:
    """CARD5(b): conservatively True when the clip's FIRST sentence already NAMES its subject
    (the query topic or the anchor's primary concept) — a grounded rapidfuzz match above a HIGH
    threshold (as context_card._grounded uses). When True an orientation card is redundant, so
    step 6 suppresses it. Only a CLEAR match suppresses, to avoid removing a needed card."""
    from ..select import _normalize
    i0 = s.get("sentence_start_idx")
    if i0 is None or i0 < 0 or i0 >= len(sentences):
        return False
    first = _normalize(sentences[i0].text or "")
    if not first:
        return False
    subjects: list[str] = []
    if topic and topic.strip():
        subjects.append(topic)
    anchor = units_by_id.get(s.get("anchor_id", ""))
    if anchor is not None and getattr(anchor, "concepts_introduced", None):
        subjects.append(anchor.concepts_introduced[0])
    from rapidfuzz import fuzz
    for subj in subjects:
        needle = _normalize(subj)
        if len(needle) >= 4 and fuzz.partial_ratio(needle, first) >= 90.0:
            return True
    return False


def _introducer_index(units) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for u in units:
        for c in u.concepts_introduced:
            idx.setdefault(c, []).append(u.unit_id)
    return idx


def effective_max_clips(settings: dict, budget: int) -> int:
    """Q1a: the final ship cap must never undercut the content-scaled anchor budget —
    UNLESS the user explicitly set max_clips (an explicit dial is respected exactly,
    even when smaller). settings["max_clips"]=None (the DEFAULTS value) means inherit:
    max(MAX_SEGMENTS, budget)."""
    configured = settings.get("max_clips")
    if configured is not None:
        return max(1, int(configured))
    return max(int(config.MAX_SEGMENTS), int(budget))


def assemble_clips(structure: Structure, topic: str, sentences: list[Sentence], url: str,
                   video_id: str, settings: dict, adapter,
                   progress: ProgressCb = None,
                   stats: Optional[dict] = None) -> tuple[list[dict], str, list[Rejection]]:
    """``stats`` (optional, additive — I1): a caller-owned dict filled with machine-readable
    run signals the eval columns need but the (human-facing) notes string can't carry:
    plan_engine ('plan' | 'plan-fallback' | 'priority'), n_arcs_detected (plan selector
    only — arc detection never runs on the priority path), anchor_budget (Q1a content-
    scaled budget), relevance_bypass (Q1d), n_refund_rounds / n_refund_clips /
    refund_rounds (Q1e), n_refund_superset_replaced (W25-F: incumbents superseded by a
    clean refund superset), and index_clamp_events (W25-A: candidates whose cached
    sentence range overran the live sentence list — a stale-structure symptom, expect 0).
    The plan selector additionally threads 'plan_proposals' + 'arcs_verified' (round 0's;
    W25-G) so callers can persist them via artifacts.write_run_artifacts.
    The return shape, payloads and notes are unchanged."""
    def emit(frac: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    if stats is None:
        stats = {}
    stats["index_clamp_events"] = 0            # W25-A clamp telemetry (counted in _assemble_one)
    rejections: list = []

    units = structure.units
    if not units or not sentences:
        return [], "No usable structure was built for this video.", rejections
    units_by_id = structure.units_by_id()
    graph = Graph(structure.dependencies.edges, units)
    introducers = _introducer_index(units)

    # 1. topic relevance -----------------------------------------------------
    # Q1d bypass: when the query IS the video's own subject ("kinematics" asked of a
    # kinematics lecture — verbatim in the detection rationale and the title), the LLM
    # relevance pass adds only re-ranking noise. Short-circuit to the same all-1.0 scores
    # the empty-topic path uses and say so in the notes.
    selection_notes: list[str] = []
    if topic and topic.strip() and topic_matches_subject(topic, structure):
        relevance = {u.unit_id: 1.0 for u in units}
        relevance_degraded = False
        stats["relevance_bypass"] = True
        selection_notes.append("topic matches video subject — selecting everything important")
        emit(0.15, "Topic matches the video's subject — selecting everything important")
    else:
        emit(0.05, "Scoring topic relevance…")
        relevance, relevance_degraded = score_topic_relevance(
            units, topic, settings, lambda f, m="": emit(0.05 + 0.10 * f, m))

    # 2. anchors — extraction plan (default) or the legacy priority sort (P3d) -----------
    # The plan path degrades to the priority sort on plan failure ("plan-fallback", noted).
    # Detected arcs enter as synthetic anchors (arc_id-keyed) and are assembled from their
    # full hull below. Q1a: the content-scaled anchor budget feeds EVERY consumer of the
    # old MAX_ANCHORS constant — the legacy selector cap, the plan validation cap,
    # enforce_plan's quota arithmetic, the refund-loop target, and (via
    # effective_max_clips) the final ship-cap truncation.
    budget = compute_anchor_budget(units, structure.content_map, structure.detection,
                                   settings, adapter=adapter, relevance=relevance)
    stats["anchor_budget"] = budget
    selector = str(settings.get("anchor_selector") or config.ANCHOR_SELECTOR)
    if selector not in ("plan", "priority"):   # misconfiguration is loud, never silent
        print(f"[assemble] unknown anchor_selector {selector!r}; "
              "using priority selection", file=sys.stderr)
        selection_notes.append(f"unknown anchor_selector '{selector}' — "
                               "priority selection used")
        selector = "priority"

    def _select(pool_units, slots: int, round_stats: dict):
        """One selection pass (round 0 or a Q1e refund round) through the CONFIGURED
        selector over ``pool_units`` under a ``slots`` budget → (anchors, arc_anchors,
        notes). Refund rounds reuse the same selector so plan/priority behavior — arc
        detection, enforcement, fallback policy — is identical in every round."""
        sel_settings = {**settings, "max_anchors": int(slots)}
        if selector == "plan":
            return select_anchors_planned(structure, pool_units, relevance, adapter,
                                          sel_settings, topic, stats=round_stats)
        round_stats["plan_engine"] = "priority"
        return select_anchors(pool_units, relevance, adapter, sel_settings), {}, []

    anchors, arc_anchors, selection_notes_p = _select(units, budget, stats)
    selection_notes.extend(selection_notes_p)
    if not anchors:
        return [], (f"“{topic}” isn’t covered enough in this video to clip." if topic
                    else "No clip-worthy moments were found in this video."), rejections

    # 3+4. one assembly round (shared by round 0 and the Q1e refund rounds) ---
    cache: dict = {}
    cache_lock = threading.Lock()

    rej_lock = threading.Lock()
    min_score = float(settings.get("min_comprehension_score", config.JUDGE_MIN_SCORE))

    def _round(round_anchors, round_arc_anchors: dict) -> tuple[list[dict], int]:
        """Steps 3–4b for ONE selection round: build/validate/repair each candidate
        (judged concurrently), snap + dedupe, then the post-snap re-judge seam. Refund
        rounds run this IDENTICAL machinery — judge gate, verdict cache, integrity
        ledger and re-judge seams all apply, and rejections land in the shared ledger.
        Returns (surviving specs, n validated candidates)."""

        # 3. assemble + validate/repair each candidate (judged concurrently) -----
        def _assemble_one(anchor):
            arc = round_arc_anchors.get(anchor.unit_id)
            if arc is not None:                # synthetic arc anchor → whole-arc candidate
                cand = build_arc_candidate(arc, graph, adapter, units, units_by_id, sentences,
                                           relevance, settings)
            else:
                cand = build_candidate(anchor, graph, adapter, units, units_by_id, sentences,
                                       relevance, settings)
            if cand is None:
                # W25-G: a build death (empty closure inline set / arc hull whose degraded
                # path also failed) used to vanish un-ledgered — the qP diagnosis had to
                # reconstruct which anchors never became candidates. Stage 'build'.
                with rej_lock:
                    rejections.append(Rejection(
                        cand_id=f"c_{anchor.unit_id}",
                        title=anchor.topic or (anchor.summary or "")[:60],
                        role=anchor.role, stage="build",
                        reason=("arc candidate build returned None" if arc is not None
                                else "candidate build returned None (empty closure)"),
                        start=float(anchor.start), end=float(anchor.end)))
                return None
            # W25-A: a build-time index clamp means this candidate's cached sentence range
            # overran the live list (stale-structure symptom) — count it whether or not the
            # candidate survives validation (the warning itself rides the spec's warnings).
            if any(w.startswith("sentence_index_clamped") for w in (cand.warnings or ())):
                with rej_lock:                 # reuse the ledger lock for the stats write
                    stats["index_clamp_events"] += 1
            cand, rejection = validate_and_repair(cand, sentences, graph, units, units_by_id, introducers,
                                                  adapter, settings, structure.visual_summary, topic,
                                                  cache, cache_lock)
            if rejection is not None:
                with rej_lock:
                    rejections.append(rejection)
            if cand is None:
                return None
            # P1c: completeness is scored under the SAME content-bound contract the judge used
            cand.completeness_score = scoring.completeness_score(
                cand.verdict, cand.contract_role or cand.role, adapter)
            cand.grounding_score = scoring.grounding_score(cand, units_by_id)
            cand.final_quality = scoring.final_quality(cand)
            return cand

        kept = []
        workers = max(1, min(int(settings.get("judge_workers", config.JUDGE_WORKERS)),
                             len(round_anchors)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_assemble_one, a) for a in round_anchors]
            for done, fut in enumerate(as_completed(futures), start=1):
                cand = fut.result()
                if cand is not None:
                    kept.append(cand)
                emit(0.15 + 0.60 * done / len(round_anchors),
                     f"Validating clip {done}/{len(round_anchors)}")

        if not kept:
            return [], 0

        # 4. snap boundaries (metadata-aware dedupe; drops ledgered). P4a: the adapter rides
        # along so dedupe's tie-break can score contract-required-element coverage. ----------
        emit(0.80, "Refining boundaries…")
        specs, snap_rejections = snap_candidates(kept, sentences, settings, units, adapter)
        rejections.extend(snap_rejections)

        # 4b. hybrid re-judge: ANY spec whose FINAL text differs from the text its verdict was
        # issued on — merged unions, min-duration extensions, start trims, max-duration caps,
        # any future mutation — is re-judged. The trigger is the text-hash difference, never a
        # warning name; specs whose text is unchanged never spend an extra judge call. A failing
        # fresh verdict routes through the SAME asymmetric gate as the repair stage: a kill needs
        # ≥1 failure reason whose quote passes containment AND survives fresh-context
        # confirmation — unverifiable concerns ship flagged, never kill. ------------------------
        surviving = []
        for s in specs:
            text = " ".join((sentences[i].text or "")
                            for i in range(s["sentence_start_idx"], s["sentence_end_idx"] + 1)).strip()
            text_hash = judged_text_hash(text)
            if text_hash == s.get("judged_text_hash"):    # exact judged text → verdict stands
                surviving.append(s)
                continue
            # P1c: the text changed since it was judged (merge/extension/trim) — REBIND the
            # contract from the units actually in the FINAL span before re-judging, so the fresh
            # verdict, the completeness gate, and final scoring share one contract.
            contract_role = choose_contract(s.get("unit_ids", []), units_by_id, adapter) \
                or s.get("role", "")
            s["contract_role"] = contract_role
            key = (frozenset(s.get("unit_ids", [])), text_hash)
            with cache_lock:
                verdict = cache.get(key)
            if verdict is None:
                verdict = judge_clip(text, contract_role, adapter,
                                     visual_summary=structure.visual_summary(s["start"], s["end"]),
                                     topic=topic, context_card=s.get("context_card", ""))
                if not verdict.error:
                    with cache_lock:
                        cache[key] = verdict
            if verdict.error:                             # outage: ship-but-flag (rubric policy)
                s["judge_error"] = True
                s["warnings"] = tuple(set(s.get("warnings") or ()) | {"unjudged"})
                surviving.append(s)
                continue
            # deterministic quote verification of the FRESH verdict's failure reasons (zero LLM):
            # feeds phantom_verdict_rate whether the spec ships or is killed at this gate.
            flags = _verify_failure_reasons(verdict, text)
            reasons = list(verdict.failure_reasons)
            verified = [f for f, ok in zip(reasons, flags) if ok]
            killing = not (verdict.topic_identifiable and verdict.purpose_identifiable
                           and verdict.source_grounded and verdict.all_references_resolved)
            outage: dict = {}
            card_rescued = False
            if killing:
                confirmed: list = []
                if verified:                     # confirm_kill runs ONLY on the kill path
                    conf = confirm_kill(text, verified, outage)
                    confirmed = [f for f, ok in zip(verified, conf) if ok]
                # ── CARD3: card-as-repair — mirror of CARD2 at the post-snap/post-merge gate.
                # If EVERY confirmed reason is prereq/reference-family, a grounded card (from
                # this spec's referential) that flips prerequisites/references on a re-judge of
                # the SAME span rescues the would-be kill. Accept-side ONLY: a groundless card
                # or a still-failing carded verdict leaves `confirmed` intact and falls through
                # to the existing kill below; it never creates a Rejection (unverified_kill=0).
                if confirmed and _card_rescuable(confirmed):
                    card, cv = _card_rescue_verdict(
                        s, text, contract_role, adapter, units_by_id, topic,
                        structure.visual_summary(s["start"], s["end"]),
                        reasons=confirmed, introducers=introducers, units=units)
                    if card and _card_clears(cv):
                        s["context_card"] = card
                        s["ship_flagged"] = False
                        verdict, killing, confirmed, card_rescued = cv, False, [], True
                        flags = _verify_failure_reasons(cv, text)
                        reasons = list(cv.failure_reasons)
                        verified = [f for f, ok in zip(reasons, flags) if ok]
                if confirmed:                    # verified AND fresh-context confirmed → kill
                    rejections.append(Rejection(
                        cand_id=s.get("cand_id", ""), title=s.get("title", ""), role=s.get("role", ""),
                        stage=("post_merge_judge" if s.get("merged") else "post_snap_judge"),
                        reason=("merged span failed hard-core judge gate" if s.get("merged")
                                else "post-snap text change failed hard-core judge gate"),
                        score=float(verdict.score), failure_kinds=[f.kind for f in confirmed],
                        final_quality=s.get("final_quality", s.get("score")), start=s["start"], end=s["end"],
                        verified_kinds=tuple(f.kind for f in verified),
                        unverified_kinds=tuple(f.kind for f, ok in zip(reasons, flags) if not ok),
                        kill_confirmed=True))
                    continue
            s["completeness_score"] = scoring.completeness_score(verdict, contract_role, adapter)
            # refresh the phantom-rate inputs so they describe the verdict that now covers the
            # shipped text (not the stale repair-stage/winner-side stats). Kind-level tuples
            # (W25-G) feed phantom_quotable_rate — which KINDS shipped unverified matters.
            s["n_failure_reasons"] = len(reasons)
            s["n_verified"] = len(verified)
            s["verified_kinds"] = tuple(f.kind for f in verified)
            s["unverified_kinds"] = tuple(f.kind for f, ok in zip(reasons, flags) if not ok)
            s["judged_text_hash"] = text_hash             # the verdict now covers the final text
            s["hard_gates_ok"] = _hard_core_ok(verdict)   # keep the stale dedupe field truthful
            s["judge_error"] = False
            drop = {"unjudged"}
            add: set = set()
            if card_rescued:                     # CARD3: a card-completed clip HAS its card
                add.add("card_completed")
            if killing:
                # nothing survived verification + confirmation → never kill on unverifiable
                # evidence: ship flagged (existing warning mechanics; scoring dock applies).
                add.add("unverified_judge_concerns")
                if outage:
                    add.add("kill_confirm_unavailable")   # confirmation outage → conservative no-kill
                s["ship_flagged"] = True
            elif is_complete(verdict, contract_role, adapter, min_score):
                # a CLEAN fresh verdict over the FINAL text supersedes a stale repair-gate flag
                # (mirrors how 'unjudged' is cleared by a successful re-judge).
                drop |= {"unverified_judge_concerns", "kill_confirm_unavailable"}
                s["ship_flagged"] = False
            s["warnings"] = tuple((set(s.get("warnings") or ()) - drop) | add)
            surviving.append(s)
        return surviving, len(kept)

    specs, n_kept = _round(anchors, arc_anchors)
    if n_kept == 0:
        return [], "No self-contained clips could be assembled for this topic.", rejections

    # 4c. Q1e refund loop (CCQGen pattern): snap/dedupe can collapse the round-0 anchors
    # far below the budget (the audited run collapsed 12 anchors → 3 specs with 63 eligible
    # units untouched). While slots remain, re-run the SAME selector restricted to the
    # anchor-eligible units no surviving spec covers and push the new candidates through
    # the identical build/judge/snap machinery; survivors append, rejections share the
    # ledger. Bounded by refund_rounds (default config.REFUND_ROUNDS=2) and a zero-yield
    # break, so a residual that never empties can't loop forever. -----------------------
    max_refunds = settings.get("refund_rounds")
    max_refunds = int(config.REFUND_ROUNDS if max_refunds is None else max_refunds)
    rel_floor = float(settings.get("anchor_rel_floor", config.ANCHOR_REL_FLOOR))
    min_dur = float(settings.get("min_clip_duration_s", config.DEFAULTS["min_clip_duration_s"]))
    refund_rounds: list[dict] = []
    n_refund_clips = 0
    stats["n_refund_superset_replaced"] = 0
    # W25-G ship floor (config constant + DEFAULTS key; behavior identical to the old inline
    # 0.45, explicit settings win) computed ONCE here so the refund loop can gate superset
    # eviction on it (W25-F review fix) and step 5 reuses the same value.
    _fl = settings.get("quality_floor")
    floor = float(config.QUALITY_FLOOR if _fl is None else _fl)

    def _covered(u) -> bool:
        s0, s1 = u.sentence_range
        return any(s["sentence_start_idx"] <= s1 and s0 <= s["sentence_end_idx"]
                   for s in specs)

    def _refund_clean(s: dict) -> bool:
        """W25-F superset-replace bar: the newcomer's FRESH verdict (repair-stage or the 4b
        re-judge, whichever covers its final text) passed the hard core, and it isn't
        shipping flagged/unjudged — an incumbent is never displaced on weaker evidence."""
        return bool(s.get("hard_gates_ok")) and not s.get("ship_flagged") \
            and not s.get("judge_error")

    def _refund_ship_worthy(s: dict) -> bool:
        """W25-F review fix: a superset may EVICT incumbents only if it is itself ship-worthy
        under the step-5 quality floor. _refund_clean gates on the JUDGE hard core, but
        final_quality (the floor gate) is a completeness/grounding/boundary/priority blend not
        computed until step 5 — a hard-core-clean-but-low-quality superset could evict shippable
        sliver incumbents and then die at the floor, shipping NOTHING for that span (a coverage
        regression this wave's refund unlock introduced). Gate eviction on the same floor so a
        below-floor superset falls through to the trim / overlap-loser path and the incumbents
        (which are above floor and would ship) survive."""
        prov = scoring.quality(
            s.get("completeness_score", 0.0), s.get("grounding_score", 0.0),
            scoring.boundary_score(s.get("warnings")), s.get("priority", 0.0))
        return prov >= floor

    def _refund_trim(s: dict, clashes: list[dict]) -> Optional[dict]:
        """W25-F: a partially-overlapping refund candidate loses only its clashing head,
        not its life — move its start past the incumbents (sentence-true, refine's
        _trim_start_after), refresh unit_ids/referential truthfully, and re-judge the
        never-judged trimmed text under a freshly bound contract. It ships ONLY on a clean
        fresh verdict (the same trust rule as the 4b seam / severed merges: a verdict is
        valid only for the exact text it was issued on); any failure — no boundary past
        the incumbents, a still-clashing result, the anchor trimmed away, a non-clean
        re-judge — returns None and the old 'refund overlap loser' rejection stands."""
        d = _trim_start_after(s, max(k["end"] for k in clashes), sentences, min_dur)
        if d is None:
            return None
        if any(d["sentence_start_idx"] <= k["sentence_end_idx"]
               and k["sentence_start_idx"] <= d["sentence_end_idx"] for k in specs):
            return None                        # an incumbent still overlaps the trimmed span
        i0, i1 = d["sentence_start_idx"], d["sentence_end_idx"]
        keep = [uid for uid in d.get("unit_ids", []) if uid in units_by_id
                and units_by_id[uid].sentence_range[0] >= i0]
        if d.get("anchor_id") and d.get("anchor_id") not in keep:
            return None                        # the trim would cut the anchor itself
        d["unit_ids"], d["referential"] = true_contents(
            keep, [tuple(r) for r in d.get("referential", [])], units, i0, i1)
        text = " ".join((sentences[i].text or "") for i in range(i0, i1 + 1)).strip()
        text_hash = judged_text_hash(text)
        contract_role = choose_contract(d.get("unit_ids", []), units_by_id, adapter) \
            or d.get("role", "")
        key = (frozenset(d.get("unit_ids", [])), text_hash)
        with cache_lock:
            verdict = cache.get(key)
        if verdict is None:
            verdict = judge_clip(text, contract_role, adapter,
                                 visual_summary=structure.visual_summary(d["start"], d["end"]),
                                 topic=topic, context_card=d.get("context_card", ""))
            if not verdict.error:
                with cache_lock:
                    cache[key] = verdict
        if verdict.error or not is_complete(verdict, contract_role, adapter, min_score):
            return None                        # not clean → the old rejection stands
        d["contract_role"] = contract_role
        d["judged_text_hash"] = text_hash      # the clean verdict covers the trimmed text
        d["hard_gates_ok"] = _hard_core_ok(verdict)
        d["judge_error"] = False
        d["ship_flagged"] = False
        flags = _verify_failure_reasons(verdict, text)
        reasons = list(verdict.failure_reasons)
        d["n_failure_reasons"] = len(reasons)
        d["n_verified"] = sum(1 for ok in flags if ok)
        d["verified_kinds"] = tuple(f.kind for f, ok in zip(reasons, flags) if ok)
        d["unverified_kinds"] = tuple(f.kind for f, ok in zip(reasons, flags) if not ok)
        d["warnings"] = tuple(set(d.get("warnings") or ())
                              - {"unjudged", "unverified_judge_concerns",
                                 "kill_confirm_unavailable"})
        d["completeness_score"] = scoring.completeness_score(verdict, contract_role, adapter)
        units_in = [units_by_id[u] for u in d.get("unit_ids", []) if u in units_by_id]
        conf = (sum(u.source_confidence for u in units_in) / len(units_in)) if units_in else 0.0
        d["grounding_score"] = max(0.0, min(1.0, conf * (1.0 if verdict.source_grounded else 0.6)))
        return d

    while specs and len(specs) < budget and len(refund_rounds) < max_refunds:
        residual = [u for u in units
                    if _anchor_eligible(u, relevance, adapter, rel_floor) and not _covered(u)]
        if not residual:
            break
        round_stats: dict = {}
        r_anchors, r_arc_anchors, r_notes = _select(residual, budget - len(specs), round_stats)
        if not r_anchors:
            break
        selection_notes.extend(r_notes)
        new_specs, _n_kept = _round(r_anchors, r_arc_anchors)
        appended = []
        for s in new_specs:
            clashes = [k for k in specs
                       if s["sentence_start_idx"] <= k["sentence_end_idx"]
                       and k["sentence_start_idx"] <= s["sentence_end_idx"]]
            if not clashes:
                appended.append(s)
                continue
            # W25-F refund unlock — round-0 mistakes are no longer permanent. (a) A CLEAN
            # newcomer that strictly CONTAINS every clashing incumbent REPLACES them (the
            # qP c0.t1 shape: a sliver locked out the superset that fixed its coverage);
            # incumbents are ledgered at 'dedupe', never silently dropped, preserving the
            # kept+rejected+merged accounting.
            contains_all = all(
                s["sentence_start_idx"] <= k["sentence_start_idx"]
                and k["sentence_end_idx"] <= s["sentence_end_idx"]
                and (s["sentence_start_idx"] < k["sentence_start_idx"]
                     or k["sentence_end_idx"] < s["sentence_end_idx"])
                for k in clashes)
            if contains_all and _refund_clean(s) and _refund_ship_worthy(s):
                for k in clashes:
                    specs.remove(k)
                    rejections.append(Rejection(
                        cand_id=k.get("cand_id", ""), title=k.get("title", ""),
                        role=k.get("role", ""), stage="dedupe",
                        reason=f"superseded by refund superset {s.get('cand_id', '?')}",
                        final_quality=k.get("final_quality", k.get("score")),
                        start=k["start"], end=k["end"]))
                    stats["n_refund_superset_replaced"] += 1
                appended.append(s)
                continue
            # (b) partial overlap: attempt a start trim past the incumbents (re-judged,
            # clean-or-nothing) BEFORE rejecting.
            trimmed = _refund_trim(s, clashes)
            if trimmed is not None:
                appended.append(trimmed)
                continue
            rejections.append(Rejection(       # duplicate of an already-shipped span
                cand_id=s.get("cand_id", ""), title=s.get("title", ""),
                role=s.get("role", ""), stage="dedupe",
                reason=f"refund overlap loser to {clashes[0].get('cand_id', '?')}",
                final_quality=s.get("final_quality", s.get("score")),
                start=s["start"], end=s["end"]))
        specs.extend(appended)
        n_refund_clips += len(appended)
        refund_rounds.append({"round": len(refund_rounds) + 1, "n_residual": len(residual),
                              "n_anchors": len(r_anchors), "n_shipped": len(appended),
                              **{k: round_stats[k] for k in ("plan_engine", "n_arcs_detected")
                                 if k in round_stats}})
        if not appended:
            break            # a zero-yield round would reselect the same residual — stop
    stats["n_refund_rounds"] = len(refund_rounds)
    stats["n_refund_clips"] = n_refund_clips
    stats["refund_rounds"] = refund_rounds

    # 5. boundary score + quality filter (drops ledgered) --------------------
    for s in specs:
        s["boundary_score"] = scoring.boundary_score(s.get("warnings"))
        s["final_quality"] = scoring.quality(
            s.get("completeness_score", 0.0), s.get("grounding_score", 0.0),
            s["boundary_score"], s.get("priority", 0.0))
    # W25-G ship floor (config constant + DEFAULTS key; behavior identical to 0.45, explicit
    # settings win) was hoisted above the refund loop so the superset-eviction gate shares it —
    # `floor` is already bound here.
    weak = [s for s in specs if s.get("final_quality", 0.0) < floor]
    specs = scoring.drop_weak(specs, floor)
    for s in weak:
        rejections.append(Rejection(cand_id=s.get("cand_id", ""), title=s.get("title", ""),
                                    role=s.get("role", ""), stage="quality_floor",
                                    reason=f"final_quality {s.get('final_quality', 0.0):.2f} < floor {floor}",
                                    final_quality=s.get("final_quality"), start=s["start"], end=s["end"]))
    specs.sort(key=lambda s: s["final_quality"], reverse=True)
    # Q1a: the ship cap never undercuts the anchor budget unless the user explicitly
    # dialed max_clips (then their number is respected exactly, even when smaller).
    cap = effective_max_clips(settings, budget)
    for s in specs[cap:]:
        rejections.append(Rejection(cand_id=s.get("cand_id", ""), title=s.get("title", ""),
                                    role=s.get("role", ""), stage="max_clips",
                                    reason=f"beyond max_clips={cap}",
                                    final_quality=s.get("final_quality"), start=s["start"], end=s["end"]))
    specs = specs[:cap]

    if not specs:
        return [], "Clips were found but none were complete enough to keep.", rejections

    # 6. context cards -------------------------------------------------------
    # Every clip gets a grounded orientation card describing what IT covers (batched, from each
    # clip's own units) — replacing the old distant-prerequisite preface that left self-contained
    # clips blank and repeated one foundational card across a prerequisite-dense video. A
    # card-as-repair card (set during judging to resolve a specific missing prerequisite) is kept.
    emit(0.90, "Writing context cards…")
    orient = generate_orientation_cards(specs, units_by_id, adapter, topic)
    for s, card in zip(specs, orient):
        if not s.get("context_card"):            # never clobber a card-as-repair card
            s["context_card"] = card
        _card_warning(s)
        if "missing_context_card" in (s.get("warnings") or ()):
            s["boundary_score"] = scoring.boundary_score(s.get("warnings"))
            s["final_quality"] = scoring.quality(
                s.get("completeness_score", 0.0), s.get("grounding_score", 0.0),
                s["boundary_score"], s.get("priority", 0.0))

    # 7. chronological sequence + prerequisite hints ------------------------
    specs = sequence_clips(specs, graph, units_by_id)

    # 7b. P4b severed pairs — the two halves of one worked example that shipped as separate
    # clips are ALWAYS linked (watch-first + 'continues clip N'); a merge is attempted only
    # under the ship cap, and kept ONLY when the union's never-judged text earns a clean
    # fresh verdict (same trust rule as the 4b seam: a verdict is valid only for the exact
    # text it was issued on — the hash is refreshed so the record stays truthful). A
    # non-clean union verdict keeps the linked pair; nothing is killed on a merge probe.
    def _severed_merge(a: dict, b: dict) -> Optional[dict]:
        m = merge_partb(a, b, units, sentences)
        # a severed pair is disjoint (gap-limited), not overlapping — relabel truthfully
        m["warnings"] = tuple((set(m.get("warnings") or ()) - {"merged_overlap"})
                              | {"merged_severed_pair"})
        text = " ".join((sentences[i].text or "")
                        for i in range(m["sentence_start_idx"], m["sentence_end_idx"] + 1)).strip()
        text_hash = judged_text_hash(text)
        contract_role = choose_contract(m.get("unit_ids", []), units_by_id, adapter) \
            or m.get("role", "")
        m["contract_role"] = contract_role
        key = (frozenset(m.get("unit_ids", [])), text_hash)
        with cache_lock:
            verdict = cache.get(key)
        if verdict is None:
            verdict = judge_clip(text, contract_role, adapter,
                                 visual_summary=structure.visual_summary(m["start"], m["end"]),
                                 topic=topic, context_card="")
            if not verdict.error:
                with cache_lock:
                    cache[key] = verdict
        if verdict.error or not is_complete(verdict, contract_role, adapter, min_score):
            return None                        # not clean → keep the linked pair instead
        m["judged_text_hash"] = text_hash      # the clean verdict covers the union text
        m["hard_gates_ok"] = _hard_core_ok(verdict)   # refreshed with the union's own verdict
        m["judge_error"] = False
        m["ship_flagged"] = False
        flags = _verify_failure_reasons(verdict, text)
        reasons = list(verdict.failure_reasons)
        m["n_failure_reasons"] = len(reasons)
        m["n_verified"] = sum(1 for ok in flags if ok)
        m["verified_kinds"] = tuple(f.kind for f, ok in zip(reasons, flags) if ok)
        m["unverified_kinds"] = tuple(f.kind for f, ok in zip(reasons, flags) if not ok)
        m["warnings"] = tuple(set(m.get("warnings") or ())
                              - {"unjudged", "unverified_judge_concerns", "kill_confirm_unavailable"})
        m["completeness_score"] = scoring.completeness_score(verdict, contract_role, adapter)
        units_in = [units_by_id[u] for u in m.get("unit_ids", []) if u in units_by_id]
        conf = (sum(u.source_confidence for u in units_in) / len(units_in)) if units_in else 0.0
        m["grounding_score"] = max(0.0, min(1.0, conf * (1.0 if verdict.source_grounded else 0.6)))
        m["context_card"] = generate_context_card(m, units_by_id, adapter, topic)
        _card_warning(m)
        m["boundary_score"] = scoring.boundary_score(m.get("warnings"))
        m["final_quality"] = scoring.quality(
            m["completeness_score"], m["grounding_score"], m["boundary_score"],
            m.get("priority", 0.0))
        if m["final_quality"] < floor:
            return None                        # honestly-rescored union falls below the
                                               # quality floor → the linked pair stands
        return m

    max_dur = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    specs = link_severed_pairs(specs, graph, units_by_id, max_dur, merge_fn=_severed_merge)
    emit(1.0, f"Assembled {len(specs)} clip(s)")

    notes = f"{len(specs)} clip(s) about “{topic}”." if topic else f"{len(specs)} clip(s)."
    if relevance_degraded and topic:
        notes += " (topic filtering degraded — clips selected by role priority)"
    for note in selection_notes:               # plan-fallback / arc-verify degradations (P3d)
        notes += f" ({note})"
    return specs, notes, rejections
