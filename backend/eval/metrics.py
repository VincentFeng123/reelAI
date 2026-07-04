"""Metrics (spec §14).

Label-free metrics (need only the predicted clips + units + sentences): the headline
comprehension rate (the clip-only judge), ends-on-period, unresolved-reference, grounding.
Gold-based metrics (need a golden file): role accuracy, anchor recall, and — when the golden
file carries creator YouTube chapters — segmentation quality (Pk / WindowDiff at topic and
chapter granularity) plus the clip straddle rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..pipeline.assemble.validate import judge_clip
from .golden import best_match, iou


@dataclass
class Report:
    video_id: str
    n_clips: int
    metrics: dict = field(default_factory=dict)


def _clip_text(spec: dict, sentences) -> str:
    i0 = spec.get("sentence_start_idx", 0)
    i1 = spec.get("sentence_end_idx", i0)
    i0 = max(0, min(i0, len(sentences) - 1))
    i1 = max(i0, min(i1, len(sentences) - 1))
    return " ".join((sentences[i].text or "") for i in range(i0, i1 + 1)).strip()


# ── label-free ───────────────────────────────────────────────────────────────
def comprehension(specs, sentences, adapter, topic, threshold=0.7) -> tuple[float, float, int, int]:
    """(mean judge score, fraction ≥ threshold, n_judged, n_error). Clip-only — the headline.
    Error verdicts (judge outage) are EXCLUDED from mean/rate so outages can never inflate
    comprehension; the caller reports them via judge_error_rate."""
    if not specs:
        return 0.0, 0.0, 0, 0
    scores, ok, n_error = [], 0, 0
    for s in specs:
        # judged under the GOVERNING contract (contract_role, P1) — the same CLIP ROLE brief
        # the pipeline's gate used; anchor-role fallback keeps legacy specs identical.
        # item 24: pass the spec's real visual_summary (aligns with judge_failures below);
        # specs without one fall back to "" exactly as before.
        v = judge_clip(_clip_text(s, sentences), _contract_role(s), adapter,
                       visual_summary=s.get("visual_summary", ""), topic=topic,
                       context_card=s.get("context_card", ""))
        if v.error:
            n_error += 1
            continue
        scores.append(v.score)
        ok += 1 if v.score >= threshold else 0
    if not scores:
        return 0.0, 0.0, 0, n_error
    return sum(scores) / len(scores), ok / len(scores), len(scores), n_error


def ends_on_period_rate(specs, sentences) -> float:
    if not specs:
        return 0.0
    good = 0
    for s in specs:
        es = next((x for x in sentences if abs(x.end - s["end"]) < 0.3), None)
        good += 1 if (es and es.ends_with_period) else 0
    return good / len(specs)


def unresolved_reference_rate(specs, units_by_id) -> float:
    """Fraction of clips assuming a required concept neither introduced in-clip nor carded."""
    if not specs:
        return 0.0
    bad = 0
    for s in specs:
        units = [units_by_id[u] for u in s.get("unit_ids", []) if u in units_by_id]
        introduced = set().union(*[set(u.concepts_introduced) for u in units]) if units else set()
        required = set().union(*[set(u.concepts_required) for u in units]) if units else set()
        card = (s.get("context_card", "") or "").lower()
        missing = [c for c in (required - introduced) if c and c not in card]
        bad += 1 if missing else 0
    return bad / len(specs)


def opening_onset_rate(specs, sentences) -> float:
    """Fraction of clips whose FIRST sentence is a discourse-onset (not mid-thought /
    not at the answer). Operationalizes 'a cold viewer isn't dropped in the middle'
    (PodReels' audience-confusion signal). The headline START-quality number."""
    from ..pipeline.discourse import opens_mid_thought
    if not specs:
        return 0.0
    good = 0
    for s in specs:
        i0 = s.get("sentence_start_idx", 0)
        i0 = max(0, min(i0, len(sentences) - 1)) if sentences else 0
        text = sentences[i0].text if sentences else ""
        good += 0 if opens_mid_thought(text) else 1
    return good / len(specs)


def grounding_ok_rate(specs) -> float:
    """Fraction of non-empty context cards (empty == omitted-not-hallucinated is fine)."""
    carded = [s for s in specs if s.get("context_card")]
    return len(carded) / len(specs) if specs else 0.0


def grounding_precision(specs, units_by_id) -> float:
    """Fraction of context-card SENTENCES entailed by one of the clip's own source units
    (anchor + referential). Independently re-verifies the card generator's grounding guarantee
    — vs. grounding_ok_rate/carded_rate which only measure card presence. NaN when no cards."""
    import re

    from ..pipeline.assemble.context_card import _grounded
    total, ok = 0, 0
    for s in specs:
        card = (s.get("context_card") or "").strip()
        if not card:
            continue
        ids = [s.get("anchor_id", "")] + [uid for uid, _rel in s.get("referential", [])]
        allowed = [units_by_id[u] for u in ids if u in units_by_id]
        if not allowed:
            continue
        for sent in (p.strip() for p in re.split(r"(?<=[.!?])\s+", card) if p.strip()):
            total += 1
            ok += 1 if any(_grounded(sent, u) for u in allowed) else 0
    return ok / total if total else float("nan")


# ── contract / prerequisite / visual / sequence (spec §14 additions) ─────────
_PROBLEM_ANCHORS = frozenset({"result", "solution", "derivation", "worked_example", "calculation",
                              "implementation", "debugging_step", "cooking_action"})


def _clip_units(spec, units_by_id):
    return [units_by_id[u] for u in spec.get("unit_ids", []) if u in units_by_id]


def _contract_role(spec) -> str:
    """The role whose contract governs the clip: the content-bound contract_role (P1) when
    the pipeline recorded one, else the anchor role (older specs / legacy runs)."""
    return spec.get("contract_role") or spec.get("role", "")


def _required_satisfied(spec, units_by_id, adapter) -> bool:
    """Every 'required' element of the clip's GOVERNING contract (contract_role, falling back
    to the anchor role) is covered by a clip unit's role."""
    c = adapter.contract_for(_contract_role(spec)) if adapter else None
    if not c:
        return True                                        # no contract → vacuously complete
    roles = {u.role for u in _clip_units(spec, units_by_id)}
    return all(any(r in roles for r in el.roles) for el in c.elements if el.necessity == "required")


def context_complete_rate(specs, units_by_id, adapter) -> float:
    """Fraction of clips whose governing contract has all required elements present in-clip."""
    if not specs:
        return 0.0
    return sum(1 for s in specs if _required_satisfied(s, units_by_id, adapter)) / len(specs)


def prerequisite_gap_rate(specs, units_by_id) -> float:
    """Fraction of clips assuming a concept not introduced in-clip, not carded, and not introduced
    by an EARLIER clip in the (chronological) sequence."""
    if not specs:
        return 0.0
    ordered = sorted(specs, key=lambda s: s.get("start", 0.0))
    introduced_so_far: set = set()
    bad = 0
    for s in ordered:
        units = _clip_units(s, units_by_id)
        intro = set().union(*[set(u.concepts_introduced) for u in units]) if units else set()
        req = set().union(*[set(u.concepts_required) for u in units]) if units else set()
        card = (s.get("context_card", "") or "").lower()
        gap = [c for c in req if c and c not in intro and c not in introduced_so_far and c not in card]
        bad += 1 if gap else 0
        introduced_so_far |= intro
    return bad / len(specs)


def worked_example_completeness(specs, units_by_id, adapter) -> float:
    """For problem-shaped anchors: required contract elements present AND in position order
    (before ≤ within ≤ after by first satisfying unit's start). NaN if no such anchors."""
    targets = [s for s in specs
               if _contract_role(s) in _PROBLEM_ANCHORS and adapter.contract_for(_contract_role(s))]
    if not targets:
        return float("nan")
    pos_rank = {"before": 0, "within": 1, "after": 2}
    ok = 0
    for s in targets:
        c = adapter.contract_for(_contract_role(s))
        units = _clip_units(s, units_by_id)
        seq, complete = [], True
        for el in c.elements:
            if el.necessity != "required":
                continue
            starts = [u.start for u in units if u.role in el.roles]
            if not starts:
                complete = False
                break
            seq.append((pos_rank.get(el.position, 1), min(starts)))
        if complete:
            # every lower-position element must start no later than every higher-position one
            # (compare ALL cross-band pairs, not just adjacent — a same-position plateau, e.g. two
            # 'within' elements, would otherwise skip the before≤after check).
            complete = all(seq[i][1] <= seq[j][1] + 1e-6
                           for i in range(len(seq)) for j in range(len(seq)) if seq[i][0] < seq[j][0])
        ok += 1 if complete else 0
    return ok / len(targets)


def visual_completeness(specs, units_by_id) -> float:
    """Fraction of clips whose declared visual_dependencies are ALL linked to a real on-screen
    event. NaN when no clip references a visual (transcript-only / no perception)."""
    relevant, ok = 0, 0
    for s in specs:
        deps = [d for u in _clip_units(s, units_by_id) for d in u.visual_dependencies]
        if not deps:
            continue
        relevant += 1
        ok += 1 if all(d.visual_event_id for d in deps) else 0
    return ok / relevant if relevant else float("nan")


def sequence_coherence(specs, units_by_id) -> float:
    """Fraction of clips (chronological) with NO forward reference: they don't require a concept
    whose earliest introducer is a LATER clip, unless it's introduced in-clip or carded."""
    if not specs:
        return 0.0
    ordered = sorted(specs, key=lambda s: s.get("start", 0.0))
    per_clip, intro_idx = [], {}
    for i, s in enumerate(ordered):
        units = _clip_units(s, units_by_id)
        intro = set().union(*[set(u.concepts_introduced) for u in units]) if units else set()
        req = set().union(*[set(u.concepts_required) for u in units]) if units else set()
        per_clip.append((intro, req, (s.get("context_card", "") or "").lower()))
        for c in intro:
            intro_idx.setdefault(c, i)                      # earliest introducer index
    ok = 0
    for i, (intro, req, card) in enumerate(per_clip):
        fwd = [c for c in req if c and c not in intro and c not in card and intro_idx.get(c, -1) > i]
        ok += 1 if not fwd else 0
    return ok / len(ordered)


# ── judge-integrity (Wave 1: phantom verdicts, ship-flags, kill split) ───────
def phantom_verdict_rate(specs, rejections) -> float:
    """Share of ALL emitted judge failure reasons that failed evidence-quote containment
    verification, across accepts and rejects — the measured rate of phantom verdicts the
    Wave-1 asymmetric gate exists to catch. Shipped specs carry n_failure_reasons/n_verified
    (recorded on every validate_and_repair outcome); judge-kill Rejections carry
    verified_kinds/unverified_kinds. Zero LLM calls. NaN when no reasons carry stats."""
    total = phantom = 0
    for s in specs or []:
        n = int(s.get("n_failure_reasons", 0) or 0)
        v = int(s.get("n_verified", 0) or 0)
        total += n
        phantom += max(0, n - v)
    for r in rejections or []:
        nv = len(getattr(r, "verified_kinds", ()) or ())
        nu = len(getattr(r, "unverified_kinds", ()) or ())
        total += nv + nu
        phantom += nu
    return phantom / total if total else float("nan")


# W25-G: the kinds a judge CAN evidence with a verbatim clip quote — the off-idea text /
# ungrounded sentence / dangling phrase IS in the span. Absence-shaped kinds
# (missing_prerequisite/missing_result/…) are unquotable by construction (the missing thing
# is, by definition, not in the transcript) and by design can never kill, so a phantom rate
# mixing them mostly measures the vocabulary, not gate health. Substring match on the
# normalized kind tolerates LLM variants ('over-inclusion', 'off_topic_content').
_QUOTABLE_KINDS = ("off_topic", "over_inclusion", "not_source_grounded", "unresolved_reference")


def _kind_quotable(kind) -> bool:
    k = re.sub(r"[^a-z0-9]+", "_", str(kind or "").lower()).strip("_")
    return any(q in k for q in _QUOTABLE_KINDS)


def phantom_quotable_rate(specs, rejections) -> float:
    """phantom_verdict_rate restricted to QUOTABLE kinds (W25-G): the share of emitted
    off_topic/over_inclusion/not_source_grounded/unresolved_reference reasons that failed
    quote verification. These reasons could have quoted the span, so a phantom here is a
    real judge-integrity failure — unlike the absence kinds dominating the mixed 0.265.
    Reads the kind-level records (spec['verified_kinds']/['unverified_kinds'], mirrors of
    the Rejection fields); count-only legacy specs contribute nothing. NaN when no
    quotable-kind reasons were recorded."""
    total = phantom = 0
    for s in specs or []:
        total += sum(1 for k in (s.get("verified_kinds") or ()) if _kind_quotable(k))
        bad = sum(1 for k in (s.get("unverified_kinds") or ()) if _kind_quotable(k))
        total += bad
        phantom += bad
    for r in rejections or []:
        total += sum(1 for k in (getattr(r, "verified_kinds", ()) or ()) if _kind_quotable(k))
        bad = sum(1 for k in (getattr(r, "unverified_kinds", ()) or ()) if _kind_quotable(k))
        total += bad
        phantom += bad
    return phantom / total if total else float("nan")


def shipped_flagged(specs) -> tuple[int, float]:
    """(count, rate) of shipped clips carrying 'unverified_judge_concerns' — clips the old
    gate would have killed but whose failure evidence survived neither quote verification
    nor confirmation. Rate is NaN when nothing shipped."""
    n = sum(1 for s in specs or [] if "unverified_judge_concerns" in (s.get("warnings") or ()))
    return n, (n / len(specs) if specs else float("nan"))


_JUDGE_KILL_STAGES = frozenset({"repair", "post_merge_judge", "post_snap_judge"})


def kill_counts(rejections) -> tuple[int, int]:
    """(verified_kill, unverified_kill): judge-gate rejections split by whether the kill was
    upheld by quote verification + fresh-context confirmation (Rejection.kill_confirmed).
    Only judge stages count — mechanical drops (snap/dedupe/quality_floor/max_clips) are not
    judge verdicts. Every judge gate (repair AND the 4b post-merge/post-snap re-judge) now
    routes kills through the same asymmetric confirmation gate, so unverified_kill should
    structurally approach 0 — it stays as the regression tripwire for any future gate drift."""
    verified = sum(1 for r in rejections or [] if getattr(r, "kill_confirmed", False))
    unverified = sum(1 for r in rejections or []
                     if getattr(r, "stage", "") in _JUDGE_KILL_STAGES
                     and not getattr(r, "kill_confirmed", False))
    return verified, unverified


def judge_failures(specs, sentences, adapter, topic):
    """Per-clip judge verdicts for --verbose: list of (role, score, [failure kinds], error).
    Judged under the governing contract (contract_role, P1) like the pipeline gate; the
    reported role stays the ANCHOR role (provenance)."""
    out = []
    for s in specs:
        v = judge_clip(_clip_text(s, sentences), _contract_role(s), adapter,
                       visual_summary=s.get("visual_summary", ""), topic=topic,
                       context_card=s.get("context_card", ""))
        out.append((s.get("role", ""), round(v.score, 2), [f.kind for f in v.failure_reasons],
                    bool(v.error)))
    return out


# ── Wave 2 extraction-quality columns (label-free; I1) ───────────────────────
def chapter_coverage(specs, units_by_id, topic_nodes) -> float:
    """Fraction of content-map TOPIC nodes with ≥1 shipped clip. A clip covers every topic
    node one of its units maps to — the same unit→topic rule plan enforcement uses
    (candidates._unit_topic_id: the unit's own node_id, else sentence-range containment).
    NaN when the map has no topic level (nothing to cover); 0.0 when topics exist but no
    clip landed in any of them."""
    if not topic_nodes:
        return float("nan")
    from ..pipeline.assemble.candidates import _unit_topic_id
    node_ids = {n.node_id for n in topic_nodes}
    covered: set = set()
    for s in specs or []:
        for uid in s.get("unit_ids", []):
            u = units_by_id.get(uid)
            if u is None:
                continue
            t = _unit_topic_id(u, topic_nodes)
            if t in node_ids:
                covered.add(t)
    return len(covered) / len(topic_nodes)


def topic_span_coverage(specs, topic_nodes) -> float:
    """W25-C sibling of chapter_coverage: SHIPPED SECONDS over NODE SECONDS across the
    content-map topic nodes (interval union of shipped spec spans clipped to each node),
    so sliver coverage is visible — the qP items-3/4 failure shipped a 24.2s clip into an
    ~88s topic that chapter_coverage scored 1.0. NaN when no topic node carries timing
    (nothing measurable); untimed nodes are excluded from both numerator and denominator."""
    nodes = [n for n in topic_nodes or [] if float(n.end) > float(n.start)]
    total = sum(float(n.end) - float(n.start) for n in nodes)
    if not nodes or total <= 0:
        return float("nan")
    shipped = 0.0
    for n in nodes:
        lo, hi = float(n.start), float(n.end)
        ivs = sorted((max(lo, float(s.get("start", 0.0))), min(hi, float(s.get("end", 0.0))))
                     for s in specs or []
                     if float(s.get("end", 0.0)) > lo and float(s.get("start", 0.0)) < hi)
        cur_lo, cur_hi = None, None
        for s0, s1 in ivs:
            if cur_hi is None or s0 > cur_hi:  # disjoint — flush the running interval
                shipped += (cur_hi - cur_lo) if cur_hi is not None else 0.0
                cur_lo, cur_hi = s0, s1
            else:
                cur_hi = max(cur_hi, s1)
        if cur_hi is not None:
            shipped += cur_hi - cur_lo
    return shipped / total


def n_arc_clips(specs) -> int:
    """Shipped clips carrying ANY detected-arc provenance — spec['arc_id'] is set by
    candidates.build_arc_candidate (both the full-hull and the degraded-closure path) and
    rides the snap/dedupe metadata; spec['arc_ids'] is the merge-union provenance list
    integrity.merge_partb adds (W25-D: a non-arc merge winner used to strip the loser's
    arc_id and undercount this metric)."""
    return sum(1 for s in specs or [] if s.get("arc_id") or s.get("arc_ids"))


def trim_moves(specs) -> int:
    """Total repair TRIM moves taken across shipped clips (P2): each judged probe of the
    trim lattice counts one move (validate_and_repair records them as spec['n_trims'])."""
    return sum(int(s.get("n_trims", 0) or 0) for s in specs or [])


def severed_pair_counts(specs) -> tuple[int, int]:
    """(linked, merged) severed worked-example pairs (P4b): ``linked`` counts the
    'continues clip N' notes stamped on the LATER half of every still-severed pair;
    ``merged`` counts clips carrying the 'merged_severed_pair' warning (a pair replaced
    by its cleanly re-judged union)."""
    linked = sum(1 for s in specs or []
                 for note in (s.get("notes") or []) if str(note).startswith("continues clip "))
    merged = sum(1 for s in specs or [] if "merged_severed_pair" in (s.get("warnings") or ()))
    return linked, merged


def edge_clean_rates(specs) -> tuple[float, float]:
    """VID2 (ADVISORY): (starts_clean_rate, ends_clean_rate) over shipped specs carrying an
    edge-probe verdict (spec['starts_clean_audio'] / ['ends_clean_audio']). NaN when NO spec
    carries a verdict — the edge probe is default-OFF and eval never runs it, so these read
    null unless a probed run is scored. Purely reported, never gating."""
    starts = [bool(s.get("starts_clean_audio")) for s in specs or []
              if s.get("starts_clean_audio") is not None]
    ends = [bool(s.get("ends_clean_audio")) for s in specs or []
            if s.get("ends_clean_audio") is not None]
    sr = sum(starts) / len(starts) if starts else float("nan")
    er = sum(ends) / len(ends) if ends else float("nan")
    return sr, er


def min_duration_extensions(specs) -> int:
    """W25-G: shipped specs whose snap EXTENDED the span to reach min duration —
    refine._snap_one marks 'extended_for_min_duration' (content beyond the judged span;
    the 4b re-judge covers it, but the count tracks how often 2s anchors get padded).
    0 is a real measurement (nothing was extended), never a NaN."""
    return sum(1 for s in specs or []
               if "extended_for_min_duration" in (s.get("warnings") or ()))


def forward_requires_edges(structure) -> int:
    """W25-G graph lint recomputed in eval (never trusts a persisted counter): 'requires'
    edges pointing FORWARD in unit order — the source presupposes a unit that hasn't
    happened yet (the pre-W25-B ``lst[0]`` future-introducer class). Expect 0 post-W25-B;
    nonzero means an LLM edge violated the direction contract or a stale cache is in
    play. Edges touching unknown unit ids are ignored (never a crash)."""
    units = getattr(structure, "units", None) or []
    order = {u.unit_id: i for i, u in enumerate(units)}
    deps = getattr(structure, "dependencies", None)
    edges = getattr(deps, "edges", None) or []
    return sum(1 for e in edges if getattr(e, "relation", "") == "requires"
               and e.source in order and e.target in order
               and order[e.source] < order[e.target])


# ── gold-based ───────────────────────────────────────────────────────────────
def role_accuracy(pred_units, gold_units) -> float:
    if not gold_units:
        return float("nan")
    matched, correct = 0, 0
    for gu in gold_units:
        pm = best_match(float(gu["start"]), float(gu["end"]),
                        [{"start": u.start, "end": u.end, "role": u.role} for u in pred_units])
        if pm:
            matched += 1
            correct += 1 if pm["role"] == gu.get("role") else 0
    return correct / matched if matched else float("nan")


def anchor_recall(gold_anchors, specs) -> float:
    if not gold_anchors:
        return float("nan")
    covered = 0
    for ga in gold_anchors:
        hit = any(iou(float(ga["start"]), float(ga["end"]), s["start"], s["end"]) >= 0.5 for s in specs)
        covered += 1 if hit else 0
    return covered / len(gold_anchors)


# W25-G standing inventory recall: the coverage-audit acceptance rule. ONE-SIDED overlap
# (overlap seconds / ITEM duration ≥ 0.60 against A SINGLE shipped spec) — deliberately not
# IoU (a big clip containing a small item covers it) and deliberately not a union (two
# slivers half-covering an item teach neither half; the qP audit counted per-clip). The
# gold file's own 'shipped' flags used the stricter ≥0.80; the standing metric is the
# ≥0.60 acceptance-protocol rule.
INVENTORY_COVER_FRAC = 0.60


def inventory_coverage(items, specs) -> list[tuple[dict, float, bool]]:
    """Per-item detail behind inventory_recall (and run_eval --verbose): [(item,
    best_overlap_fraction, covered)] in gold-file order. A zero/negative-duration item can
    never be covered (fraction 0.0, no div-by-zero); the epsilon absorbs float noise so an
    exactly-0.60 overlap counts covered."""
    out = []
    for it in items or []:
        s, e = float(it.get("start", 0.0)), float(it.get("end", 0.0))
        dur = e - s
        best = 0.0
        if dur > 0:
            for sp in specs or []:
                ov = min(e, float(sp.get("end", 0.0))) - max(s, float(sp.get("start", 0.0)))
                if ov > 0:
                    best = max(best, ov / dur)
        out.append((it, best, best >= INVENTORY_COVER_FRAC - 1e-9))
    return out


def inventory_recall(items, specs) -> float:
    """Fraction of gold 'inventory' items covered by a shipped spec under the one-sided
    ≥0.60 rule. NaN when the gold file carries no items (a video without an inventory must
    not read as 0 recall and drag the aggregate)."""
    cov = inventory_coverage(items, specs)
    if not cov:
        return float("nan")
    return sum(1 for _it, _f, ok in cov if ok) / len(cov)


def boundary_error(gold_anchors, specs) -> float:
    """Mean boundary offset (seconds) of predicted clips vs their best-matching gold anchor:
    mean(|Δstart| + |Δend|) over clips that match a gold span (IoU≥0.5). NaN when nothing matches."""
    if not gold_anchors or not specs:
        return float("nan")
    gold_items = [{"start": float(g["start"]), "end": float(g["end"])} for g in gold_anchors]
    errs = []
    for s in specs:
        m = best_match(float(s["start"]), float(s["end"]), gold_items)
        if m:
            errs.append(abs(float(s["start"]) - m["start"]) + abs(float(s["end"]) - m["end"]))
    return sum(errs) / len(errs) if errs else float("nan")


# ── segmentation gold (creator YouTube chapters; YTSeg protocol, EACL 2024) ──
def _seg_bounds(seg: dict) -> tuple[float, float]:
    """(start, end) seconds of a segment dict — accepts both the normalized {start,end} form
    and yt-dlp's raw {start_time,end_time} form."""
    s = float(seg.get("start", seg.get("start_time", 0.0)))
    e = float(seg.get("end", seg.get("end_time", s)))
    return s, e


def assign_segments(sentences, segments: list[dict]) -> list[int]:
    """Label each sentence with the index of the segment sharing the greatest TIME OVERLAP —
    the YTSeg reference-construction rule (Retkowski & Waibel, EACL 2024: creator chapters are
    segmentation gold; chapter timestamps map to sentence boundaries by greatest overlap).
    Ties break to the EARLIER segment. A sentence overlapping no segment (a gap, or zero
    duration) gets the segment nearest its midpoint. Segments must be time-ordered.
    Returns [] when there are no segments (or no sentences)."""
    if not segments:
        return []
    bounds = [_seg_bounds(s) for s in segments]
    labels = []
    for sent in sentences:
        ss, se = float(sent.start), float(sent.end)
        best_i, best_ov = 0, 0.0
        for i, (gs, ge) in enumerate(bounds):
            ov = max(0.0, min(se, ge) - max(ss, gs))
            if ov > best_ov:                       # strict > → earlier segment wins ties
                best_i, best_ov = i, ov
        if best_ov <= 0.0:
            mid = (ss + se) / 2.0
            best_i = min(range(len(bounds)),
                         key=lambda i: (max(bounds[i][0] - mid, mid - bounds[i][1], 0.0), i))
        labels.append(best_i)
    return labels


def window_size(ref_labels: list) -> int:
    """The standard Pk/WindowDiff window: k = max(2, round(mean TRUE segment length in
    sentences / 2)) — half the mean reference segment size (Beeferman et al. 1999 convention,
    kept by Pevzner & Hearst 2002). Segments are counted as contiguous label runs."""
    n = len(ref_labels)
    if n == 0:
        return 2
    n_segments = 1 + sum(1 for i in range(n - 1) if ref_labels[i] != ref_labels[i + 1])
    return max(2, round(n / n_segments / 2))


def pk(ref_labels: list, hyp_labels: list, k: int | None = None) -> float:
    """Pk (Beeferman, Berger & Lafferty 1999, "Statistical Models for Text Segmentation",
    Machine Learning 34): slide a probe window of width k over the sentence sequence; at each
    position i check whether sentences i and i+k fall in the SAME segment under the reference
    vs the hypothesis; Pk is the fraction of probes where the two disagree (0 = perfect;
    lower is better). k defaults to window_size(ref_labels). NaN when the label sequences are
    empty, differ in length, or are too short for the window (n <= k)."""
    n = len(ref_labels)
    if n == 0 or n != len(hyp_labels):
        return float("nan")
    if k is None:
        k = window_size(ref_labels)
    if n <= k:
        return float("nan")
    disagree = sum(
        1 for i in range(n - k)
        if (ref_labels[i] == ref_labels[i + k]) != (hyp_labels[i] == hyp_labels[i + k]))
    return disagree / (n - k)


def windowdiff(ref_labels: list, hyp_labels: list, k: int | None = None) -> float:
    """WindowDiff (Pevzner & Hearst 2002, "A Critique and Improvement of an Evaluation Metric
    for Text Segmentation", Computational Linguistics 28(1)): slide a window of width k; at
    each position i count the boundaries falling inside the window in the reference and the
    hypothesis, and penalize whenever the two counts differ. Fixes Pk's under-penalization of
    false positives and near-misses. Same k default and NaN conventions as pk()."""
    n = len(ref_labels)
    if n == 0 or n != len(hyp_labels):
        return float("nan")
    if k is None:
        k = window_size(ref_labels)
    if n <= k:
        return float("nan")

    def nb(labels, i):                             # boundaries between positions i .. i+k
        return sum(1 for j in range(i, i + k) if labels[j] != labels[j + 1])

    err = sum(1 for i in range(n - k) if nb(ref_labels, i) != nb(hyp_labels, i))
    return err / (n - k)


def clip_straddle_rate(specs, chapters: list[dict], tol: float = 5.0) -> float:
    """Fraction of shipped clips whose [start,end] crosses an internal gold-chapter boundary
    with MORE than `tol` seconds of the clip on BOTH sides of it — a clip genuinely split
    across two chapters, not one merely grazing a boundary. NaN when there are no clips or
    no gold chapters; 0.0 when chapters exist but nothing straddles (e.g. a single chapter)."""
    if not specs or not chapters:
        return float("nan")
    starts = sorted(_seg_bounds(c)[0] for c in chapters)
    boundaries = starts[1:]                        # internal boundaries only
    hit = sum(1 for s in specs
              if any(float(s["start"]) + tol < b < float(s["end"]) - tol for b in boundaries))
    return hit / len(specs)
