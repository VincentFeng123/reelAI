"""Corruption-based judge probe (LLMBar pattern, arXiv:2310.07641) — measure per-gate
TPR/TNR of the clip-only judge with ZERO human labels.

Usage:
    python -m backend.eval.judge_probe <video_id> [...] [--limit N] [--dry-run]
    python -m backend.eval.judge_probe --all [--limit N] [--dry-run]

Video ids are REQUIRED unless --all is passed explicitly — a bare invocation must never
silently run a full LIVE probe over every cached video.

For every requested video that has BOTH a cached ``work/<id>/transcript.json`` and a cached
``work/<id>/structure.json`` (the probe never builds structure or touches the network), the
probe mirrors run_eval's frozen drive path — rebuild the sentence index, assemble clips from
the cached Structure, take the ACCEPTED specs — then manufactures deterministic known-bad
variants of each accepted clip's text:

    chop_start          remove the first sentence  → forces a mid-idea open
    chop_end            remove the last sentence   → forces a missing conclusion
    antecedent_removal  remove the sentences of an in-clip unit that an in-clip Reference
                        points back at (source_unit) → a genuinely dangling reference;
                        SKIPPED when the clip contains no such internal reference
    offtopic_splice     insert 2 contiguous sentences from a unit of a DIFFERENT
                        content-map topic (deterministically: the temporally farthest such
                        unit) into the middle of the clip

Live mode judges the original + every variant through ``validate.judge_clip`` (temperature
0.0 and judge-model routing both live inside judge_clip) and reports:

    TPR (per corruption type): fraction of corrupted variants FLAGGED — a relevant verdict
        gate came back false OR the score fell below the config threshold
        (``min_comprehension_score`` / ``JUDGE_MIN_SCORE``).
    TNR: fraction of the original (pipeline-accepted) clips that still pass. NOTE: originals
        are held to GATE_UNION (all probe gates) + the score threshold — STRICTER than the
        pipeline's role-specific acceptance (adapter.required_verdict_fields), so reported
        TNR is a lower bound on "original still passes its own acceptance bar"; the
        row-level gates JSON permits re-slicing with either rule.
    per-gate rates: how often each verdict boolean held true on originals vs came back
        false on the corruptions for which it is a relevant gate.

A compact table is printed and full row-level results are written to
``backend/eval/probe_results/probe_<timestamp>.json`` (timestamp from ``datetime`` in the
CLI ``main`` only — library functions take an explicit stamp so tests stay deterministic).

``--dry-run`` prints every corruption variant WITHOUT judging, fully offline: the LLM calls
inside assembly are replaced with a deterministic perfect-pass stub (every judge verdict is
a 10/10 pass, every other schema gets its defaults), so every assembled candidate is treated
as accepted and ZERO ``llm_json`` calls are made.

Two deliberate deviations from run_eval's drive path, both because the probe only ever
consumes cached structures: the adapter is resolved from the cached ``structure.detection``
domain instead of a fresh ``detect_content_type`` LLM call, and perception is never computed.
Judging mirrors the eval comprehension metric exactly (``visual_summary=""``, the spec's own
``context_card``), held constant between the original and its variants.

IMPORTANT — treat probe TPR as an UPPER BOUND on the judge's real-world sensitivity:
mechanical corruptions (a chopped sentence, a spliced far-away passage) are blunter and
easier to notice than the natural failures the judge must catch in production (subtle
mid-clause cuts, soft topic drift, implicit prerequisites). A high probe TPR does not prove
the judge catches natural failures; a low probe TPR is strong evidence it cannot.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from .. import config
from ..adapters import get_adapter
from ..pipeline.assemble import assemble_clips
from ..pipeline.assemble.validate import JudgeVerdict, judge_clip
from ..pipeline.understand import load_structure
from .golden import load_golden
from .run_eval import _sentences

PROBE_RESULTS_DIR = Path(__file__).resolve().parent / "probe_results"

CORRUPTION_KINDS = ("chop_start", "chop_end", "antecedent_removal", "offtopic_splice")

# The verdict booleans each corruption is designed to trip ("relevant gates"). A corrupted
# variant counts as FLAGGED when any relevant gate is false OR score < threshold.
RELEVANT_GATES: dict[str, tuple[str, ...]] = {
    "chop_start": ("source_grounded", "all_references_resolved", "topic_identifiable"),
    "chop_end": ("result_complete", "reasoning_complete", "purpose_identifiable"),
    "antecedent_removal": ("all_references_resolved", "prerequisites_satisfied"),
    "offtopic_splice": ("topic_identifiable", "purpose_identifiable", "source_grounded"),
}
GATE_UNION: tuple[str, ...] = tuple(sorted({g for gs in RELEVANT_GATES.values() for g in gs}))


@dataclass(frozen=True)
class Corruption:
    kind: str                      # one of CORRUPTION_KINDS
    text: str                      # the corrupted clip transcript
    detail: str                    # human-readable description of what was changed
    relevant_gates: tuple[str, ...]


# ── pure corruption generators (deterministic; no LLM, no I/O) ────────────────
def _clip_range(spec: dict, n_sents: int) -> tuple[int, int]:
    """Clamped inclusive sentence range of a clip spec (mirrors metrics._clip_text)."""
    i0 = int(spec.get("sentence_start_idx", 0))
    i1 = int(spec.get("sentence_end_idx", i0))
    i0 = max(0, min(i0, n_sents - 1))
    i1 = max(i0, min(i1, n_sents - 1))
    return i0, i1


def _text(indices, sentences) -> str:
    return " ".join((sentences[i].text or "") for i in indices).strip()


def corrupt_chop_start(spec: dict, sentences) -> Optional[Corruption]:
    """Remove the clip's first sentence — a mid-idea open. None for 1-sentence clips."""
    i0, i1 = _clip_range(spec, len(sentences))
    if i1 <= i0:
        return None
    return Corruption("chop_start", _text(range(i0 + 1, i1 + 1), sentences),
                      f"removed first sentence [{i0}]", RELEVANT_GATES["chop_start"])


def corrupt_chop_end(spec: dict, sentences) -> Optional[Corruption]:
    """Remove the clip's last sentence — a missing conclusion. None for 1-sentence clips."""
    i0, i1 = _clip_range(spec, len(sentences))
    if i1 <= i0:
        return None
    return Corruption("chop_end", _text(range(i0, i1), sentences),
                      f"removed last sentence [{i1}]", RELEVANT_GATES["chop_end"])


def corrupt_antecedent_removal(spec: dict, sentences, units_by_id: dict) -> Optional[Corruption]:
    """Remove the sentences of an in-clip unit that an in-clip Reference points back at,
    leaving the referring phrase genuinely dangling. Deterministic: the FIRST such reference
    in the spec's (chronological) unit order wins. Returns None — the skip rule — when no
    clip unit carries a Reference whose source_unit is also inside the clip (or when removal
    would empty the clip)."""
    i0, i1 = _clip_range(spec, len(sentences))
    unit_ids = [u for u in spec.get("unit_ids", []) if u in units_by_id]
    in_clip = set(unit_ids)
    for uid in unit_ids:
        for ref in units_by_id[uid].references:
            src = ref.source_unit
            if not src or src == uid or src not in in_clip:
                continue
            r0, r1 = units_by_id[src].sentence_range
            r0, r1 = max(int(r0), i0), min(int(r1), i1)
            if r1 < r0:
                continue                       # source unit's sentences fall outside the span
            keep = [i for i in range(i0, i1 + 1) if not (r0 <= i <= r1)]
            text = _text(keep, sentences)
            if not text:
                continue                       # removal would empty the clip — not a variant
            return Corruption(
                "antecedent_removal", text,
                f"removed source unit {src} (sentences {r0}-{r1}), "
                f"dangling reference {ref.text!r} in {uid}",
                RELEVANT_GATES["antecedent_removal"])
    return None


def _topic_of(unit, topics) -> Optional[str]:
    """The content-map topic node a unit belongs to: its own node_id when that is a topic
    node, else the topic whose time span contains the unit midpoint. None if unassignable."""
    if unit.node_id and any(t.node_id == unit.node_id for t in topics):
        return unit.node_id
    mid = (float(unit.start) + float(unit.end)) / 2.0
    for t in topics:
        if float(t.start) <= mid <= float(t.end):
            return t.node_id
    return None


def corrupt_offtopic_splice(spec: dict, sentences, structure) -> Optional[Corruption]:
    """Insert 2 contiguous sentences from a unit of a DIFFERENT content-map topic into the
    middle of the clip. Deterministic donor: the temporally farthest off-topic unit (unit
    midpoint vs clip midpoint; ties break to the greater unit_id). None when the clip's
    topic is unknown or no off-topic unit exists."""
    i0, i1 = _clip_range(spec, len(sentences))
    units_by_id = structure.units_by_id()
    topics = structure.content_map.topics()
    if not topics:
        return None
    clip_units = [units_by_id[u] for u in spec.get("unit_ids", []) if u in units_by_id]
    clip_topics = {t for t in (_topic_of(u, topics) for u in clip_units) if t}
    if not clip_topics:
        return None
    in_clip = {u.unit_id for u in clip_units}
    donors = [u for u in structure.units
              if u.unit_id not in in_clip
              and (_topic_of(u, topics) not in (None, *clip_topics))]
    if not donors:
        return None
    clip_mid = (float(sentences[i0].start) + float(sentences[i1].end)) / 2.0
    donor = max(donors,
                key=lambda u: (abs((float(u.start) + float(u.end)) / 2.0 - clip_mid), u.unit_id))
    s0 = max(0, min(int(donor.sentence_range[0]), len(sentences) - 1))
    s1 = max(s0, min(int(donor.sentence_range[1]), len(sentences) - 1, s0 + 1))  # ≤2 sentences
    local = list(range(i0, i1 + 1))
    pos = max(1, len(local) // 2)              # middle of the clip (after sentence 1 minimum)
    indices = local[:pos] + list(range(s0, s1 + 1)) + local[pos:]
    return Corruption(
        "offtopic_splice", _text(indices, sentences),
        f"spliced sentences {s0}-{s1} from unit {donor.unit_id} "
        f"(topic {_topic_of(donor, topics)}) at clip position {pos}",
        RELEVANT_GATES["offtopic_splice"])


def corruptions_for(spec: dict, sentences, structure) -> list[Corruption]:
    """All applicable corruption variants for one accepted clip spec (inapplicable kinds —
    e.g. antecedent_removal with no internal reference — are skipped, not faked)."""
    units_by_id = structure.units_by_id()
    out = [corrupt_chop_start(spec, sentences),
           corrupt_chop_end(spec, sentences),
           corrupt_antecedent_removal(spec, sentences, units_by_id),
           corrupt_offtopic_splice(spec, sentences, structure)]
    return [c for c in out if c is not None]


# ── detection rule + aggregation (pure) ───────────────────────────────────────
def is_flagged(verdict: JudgeVerdict, relevant_gates, threshold: float) -> bool:
    """A verdict flags a clip when a relevant gate is false OR the score is below threshold."""
    if float(verdict.score) < threshold:
        return True
    return any(not bool(getattr(verdict, g, True)) for g in relevant_gates)


def _row(video_id: str, spec: dict, kind: str, verdict: JudgeVerdict, relevant_gates,
         threshold: float, detail: str = "") -> dict:
    return {
        "video_id": video_id,
        "cand_id": spec.get("cand_id", ""),
        "role": spec.get("role", ""),
        "start": spec.get("start"),
        "end": spec.get("end"),
        "kind": kind,                          # "original" | a CORRUPTION_KINDS entry
        "detail": detail,
        "error": bool(verdict.error),
        "score": round(float(verdict.score), 3),
        "gates": {g: bool(getattr(verdict, g, True)) for g in GATE_UNION},
        "relevant_gates": list(relevant_gates),
        "failure_kinds": [f.kind for f in verdict.failure_reasons],
        # None on a judge outage — error verdicts never count as detections or misses.
        "flagged": None if verdict.error else is_flagged(verdict, relevant_gates, threshold),
    }


def judge_rows(video_id: str, specs: list[dict], sentences, structure, adapter, topic: str,
               threshold: float) -> list[dict]:
    """LIVE: judge each accepted clip (original) + its corruption variants via judge_clip
    (temperature 0.0 / judge-model routing are inside judge_clip). Mirrors the eval
    comprehension call: visual_summary="", the spec's own context_card (held constant
    between original and variants so only the corruption differs)."""
    rows: list[dict] = []
    for spec in specs:
        i0, i1 = _clip_range(spec, len(sentences))
        card = spec.get("context_card", "") or ""
        # judge under the GOVERNING contract (contract_role, P1) — the same brief the
        # pipeline's gate used; the row keeps the anchor role for provenance (_row).
        role = spec.get("contract_role") or spec.get("role", "")
        v = judge_clip(_text(range(i0, i1 + 1), sentences), role, adapter,
                       visual_summary="", topic=topic, context_card=card)
        rows.append(_row(video_id, spec, "original", v, GATE_UNION, threshold))
        for c in corruptions_for(spec, sentences, structure):
            v = judge_clip(c.text, role, adapter,
                           visual_summary="", topic=topic, context_card=card)
            rows.append(_row(video_id, spec, c.kind, v, c.relevant_gates, threshold,
                             detail=c.detail))
    return rows


def summarize_probe(rows: list[dict], threshold: float) -> dict:
    """Per-corruption TPR + original TNR + per-gate true/false rates. Judge-outage rows
    (error=True) are excluded from every rate (mirrors metrics.comprehension) and reported
    as counts."""
    originals = [r for r in rows if r["kind"] == "original"]
    variants = [r for r in rows if r["kind"] != "original"]
    o_ok = [r for r in originals if not r["error"]]
    v_ok = [r for r in variants if not r["error"]]
    passed = sum(1 for r in o_ok if not r["flagged"])
    per_corruption = {}
    for kind in CORRUPTION_KINDS:
        sub = [r for r in v_ok if r["kind"] == kind]
        flagged = sum(1 for r in sub if r["flagged"])
        per_corruption[kind] = {
            "n": len(sub),
            "n_flagged": flagged,
            "tpr": (flagged / len(sub)) if sub else None,
            "mean_score": (sum(r["score"] for r in sub) / len(sub)) if sub else None,
        }
    per_gate = {}
    for g in GATE_UNION:
        rel = [r for r in v_ok if g in r["relevant_gates"]]
        per_gate[g] = {
            "original_true_rate": (sum(1 for r in o_ok if r["gates"][g]) / len(o_ok)) if o_ok else None,
            "corrupted_false_rate": (sum(1 for r in rel if not r["gates"][g]) / len(rel)) if rel else None,
            "n_corrupted_relevant": len(rel),
        }
    return {
        "threshold": threshold,
        "n_originals": len(o_ok),
        "n_originals_passed": passed,
        "n_original_errors": len(originals) - len(o_ok),
        "tnr": (passed / len(o_ok)) if o_ok else None,
        "n_variants": len(v_ok),
        "n_variant_errors": len(variants) - len(v_ok),
        "mean_original_score": (sum(r["score"] for r in o_ok) / len(o_ok)) if o_ok else None,
        "per_corruption": per_corruption,
        "per_gate": per_gate,
    }


def write_results(payload: dict, stamp: str, out_dir: Path = PROBE_RESULTS_DIR) -> Path:
    """Write the probe JSON as probe_<stamp>.json. The stamp is supplied by the caller —
    main() uses datetime; tests pass a fixed string."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"probe_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ── reporting ─────────────────────────────────────────────────────────────────
def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def print_summary(summary: dict) -> None:
    print(f"\n=== judge probe (threshold {summary['threshold']:.2f}) — "
          f"TPR is an UPPER BOUND (mechanical corruptions are easy mode) ===")
    print(f"originals: n={summary['n_originals']}  passed={summary['n_originals_passed']}  "
          f"TNR={_fmt(summary['tnr'])}  judge_errors={summary['n_original_errors']}  "
          f"mean_score={_fmt(summary['mean_original_score'])}")
    print(f"  {'corruption':20} {'n':>4} {'flagged':>8} {'TPR':>7} {'mean_score':>11}")
    for kind in CORRUPTION_KINDS:
        c = summary["per_corruption"][kind]
        print(f"  {kind:20} {c['n']:>4} {c['n_flagged']:>8} {_fmt(c['tpr']):>7} "
              f"{_fmt(c['mean_score']):>11}")
    print(f"  {'gate':28} {'orig_true':>9} {'corrupt_false':>13} {'n_rel':>6}")
    for g in GATE_UNION:
        p = summary["per_gate"][g]
        print(f"  {g:28} {_fmt(p['original_true_rate']):>9} "
              f"{_fmt(p['corrupted_false_rate']):>13} {p['n_corrupted_relevant']:>6}")


# ── drive path (mirrors run_eval's frozen-structure path; cached inputs only) ──
def _dry_run_llm_stub(system, user, schema, **kwargs):
    """Deterministic stand-in for llm_json during --dry-run assembly: every judge verdict
    is a perfect pass (all candidates treated as accepted) and every other schema gets its
    defaults (relevance → neutral 0.5, context cards → extractive fallback). Zero LLM."""
    if schema is JudgeVerdict:
        return JudgeVerdict(reasoning="dry-run stub (never judged)", score_10=10,
                            understandable=True)
    return schema()


def load_probe_inputs(video_id: str) -> Optional[dict]:
    """Cached transcript + structure → sentences/topic/settings/adapter. None (with a skip
    note) when either cache is missing — the probe never builds or downloads anything.
    Freshness (W25-A): the live sentence index is built FIRST and threaded into
    load_structure; since the probe can only hold the cache (never rebuild), it uses the
    explicit allow_stale override — a stale structure still probes, but load_structure
    warns loudly on stderr instead of silently mixing sentence universes."""
    tpath = config.WORK_DIR / video_id / "transcript.json"
    if not tpath.exists():
        print(f"  [skip] {video_id}: no cached transcript")
        return None
    transcript = json.loads(tpath.read_text())
    transcript.setdefault("title", "")
    sents = _sentences(transcript, video_id)
    st = load_structure(video_id, sents, allow_stale=True)
    if st is None:
        print(f"  [skip] {video_id}: no cached structure.json (run the pipeline or eval first)")
        return None
    gold = load_golden(video_id) or {}
    topic = (gold.get("topics") or [""])[0]
    settings = dict(config.DEFAULTS)
    adapter = get_adapter(st.detection.domain)   # cached detection — no re-detect LLM call
    return {"structure": st, "sents": sents, "topic": topic,
            "settings": settings, "adapter": adapter}


def assemble_accepted(video_id: str, ctx: dict, dry_run: bool) -> list[dict]:
    """Assemble clips exactly like run_eval (same assemble_clips call). In --dry-run the
    llm_json entry point is patched to the deterministic stub for the duration of assembly
    so the whole path is offline."""
    url = f"https://youtu.be/{video_id}"
    args = (ctx["structure"], ctx["topic"], ctx["sents"], url, video_id,
            ctx["settings"], ctx["adapter"])
    if not dry_run:
        specs, _notes, _rejections = assemble_clips(*args)
        return specs
    with patch("backend.llm.llm_json", _dry_run_llm_stub):
        specs, _notes, _rejections = assemble_clips(*args)
    return specs


def probe_video(video_id: str, *, limit: Optional[int] = None, dry_run: bool = False,
                threshold: Optional[float] = None) -> Optional[dict]:
    """Probe one video. Dry-run prints the variants (no judging) and returns rows=[];
    live mode returns the judged rows for aggregation."""
    if threshold is None:
        threshold = float(config.DEFAULTS.get("min_comprehension_score", config.JUDGE_MIN_SCORE))
    ctx = load_probe_inputs(video_id)
    if ctx is None:
        return None
    specs = assemble_accepted(video_id, ctx, dry_run)
    if limit is not None:
        specs = specs[:limit]
    if not specs:
        print(f"  [skip] {video_id}: no accepted clips to corrupt")
        return None
    if dry_run:
        for spec in specs:
            i0, i1 = _clip_range(spec, len(ctx["sents"]))
            print(f"  clip {spec.get('cand_id', '?')} [{spec.get('role', '-')}] "
                  f"sentences {i0}-{i1}")
            variants = corruptions_for(spec, ctx["sents"], ctx["structure"])
            if not variants:
                print("    (no corruption variants)")
            for c in variants:
                print(f"    -- {c.kind}: {c.detail}")
                print(f"       {c.text}")
        return {"video_id": video_id, "n_clips": len(specs), "rows": []}
    rows = judge_rows(video_id, specs, ctx["sents"], ctx["structure"], ctx["adapter"],
                      ctx["topic"], threshold)
    n_var = sum(1 for r in rows if r["kind"] != "original")
    print(f"  {video_id}: judged {len(specs)} original(s) + {n_var} variant(s)")
    return {"video_id": video_id, "n_clips": len(specs), "rows": rows}


# ── CLI ───────────────────────────────────────────────────────────────────────
def _positive_int(value: str) -> int:
    n = int(value)                              # ValueError → argparse's clean invalid-value error
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def parse_probe_args(argv: list[str]) -> argparse.Namespace:
    """Strict argparse CLI (unknown flags and bad values ERROR instead of being ignored).
    Video ids are required positionals unless --all is passed explicitly — zero ids must
    never silently become a full live probe over every cached video."""
    p = argparse.ArgumentParser(
        prog="python -m backend.eval.judge_probe",
        description="Corruption-based judge probe over cached videos. LIVE judge calls "
                    "unless --dry-run. Requires explicit video ids, or --all to opt in to "
                    "probing every video with a cached structure.json.")
    p.add_argument("video_ids", nargs="*", metavar="video_id",
                   help="video id(s) with cached work/<id>/transcript.json + structure.json")
    p.add_argument("--all", action="store_true",
                   help="probe EVERY video with a cached structure.json (explicit opt-in)")
    p.add_argument("--limit", type=_positive_int, default=None, metavar="N",
                   help="cap the accepted clips probed per video")
    p.add_argument("--dry-run", action="store_true",
                   help="print corruption variants without judging (zero LLM calls)")
    args = p.parse_args(argv)
    if args.all and args.video_ids:
        p.error("give explicit video ids OR --all, not both")
    if not args.all and not args.video_ids:
        p.error("no video ids given (pass ids, or --all to probe every cached video)")
    return args


def main(argv: list[str]) -> None:
    a = parse_probe_args(argv)
    vids = a.video_ids or sorted(               # empty only with --all (parse enforces this)
        Path(p).parent.name for p in glob.glob(str(config.WORK_DIR / "*/structure.json")))
    if not vids:
        print("no videos with a cached structure.json under work/")
        return
    threshold = float(config.DEFAULTS.get("min_comprehension_score", config.JUDGE_MIN_SCORE))
    mode = "dry-run (print variants, no judging)" if a.dry_run else "live"
    print(f"judge probe: {len(vids)} video(s), mode={mode}, threshold={threshold}")

    all_rows: list[dict] = []
    probed: list[str] = []
    for vid in vids:
        print(f"→ {vid} …")
        try:
            res = probe_video(vid, limit=a.limit, dry_run=a.dry_run, threshold=threshold)
        except Exception as e:  # noqa: BLE001 — one bad video must not sink the probe
            print(f"  [error] {vid}: {e}")
            res = None
        if res:
            probed.append(res["video_id"])
            all_rows.extend(res["rows"])

    if a.dry_run or not all_rows:
        return
    summary = summarize_probe(all_rows, threshold)
    print_summary(summary)
    from datetime import datetime               # CLI-only: tests pass write_results a stamp
    now = datetime.now()
    payload = {"generated_at": now.isoformat(timespec="seconds"), "threshold": threshold,
               "videos": probed, "summary": summary, "rows": all_rows}
    path = write_results(payload, now.strftime("%Y%m%d-%H%M%S"))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
