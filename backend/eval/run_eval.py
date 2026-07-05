"""Run the structure-first pipeline over cached videos and report metrics (spec §14).

Usage:
    python -m backend.eval.run_eval                       # all cached transcripts
    python -m backend.eval.run_eval 1bH_ukYn81c           # specific video(s)
    python -m backend.eval.run_eval 1bH_ukYn81c --topic "the derivative"

Trustworthy-eval flags (make comprehension deltas beat LLM run-to-run noise):
    --runs N          repeat the measurement N times; report mean ± std per metric
    --freeze          reuse the cached work/<id>/structure.json instead of rebuilding it
                      (isolates assembly + judge variance — structure held constant)
    --freeze-specs    also assemble the clips once and only re-judge each run
                      (isolates judge-only variance; implies --freeze)
    --rebuild         force a fresh structure build + re-save even if a cache exists
                      (refresh the frozen cache after an understanding-stage change)

Uses cached work/<id>/transcript.json (no network). Label-free metrics run on every video;
gold metrics run additionally when eval/golden/<id>.json exists. The headline number is
comprehension_rate — the % of clips the clip-only judge deems understandable in isolation.

Rebuilding the structure every run is the dominant noise source (all the understanding-stage
LLM calls re-run). --freeze reuses the cached Structure so an A/B on a downstream change is not
fighting that noise; --runs N + the reported std tells you whether a comprehension delta exceeds
the residual judge noise. A typical A/B: `--runs 3` (baseline noise), then `--freeze --runs 3`
before vs after your change.
"""
from __future__ import annotations

import glob
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional

from .. import config
from ..adapters import select_adapter
from ..pipeline.assemble import assemble_clips
from ..pipeline.assemble.artifacts import write_run_artifacts
from ..pipeline.punctuation.service import build_sentences
from ..pipeline.understand import (
    build_structure, load_structure, save_structure, structure_is_stale,
)
from . import metrics
from .golden import gold_chapters, load_golden
from .metrics import topic_selectivity, window_len_stats


def _sentences(transcript, video_id=None):
    """The SAME punctuation-restored sentence index the orchestrator builds (W25-A): eval
    previously used the legacy pysbd path (322 vs 183 sentences on qP) and its saved
    structures poisoned the shared app cache. build_sentences is sync (the orchestrator
    merely wraps it in an executor) — called with default settings, no progress callback;
    video_id keys the punctuation chunk cache so cached videos stay offline."""
    return build_sentences(transcript, video_id, dict(config.DEFAULTS), None)


def _maybe_perceive(video_id, transcript, sents, settings):
    """Load/compute perception ONLY if the video is already cached (never a network download in eval)."""
    if not config.MULTIMODAL or str(settings.get("analysis_profile", "full")) != "full":
        return None
    vdir = config.WORK_DIR / video_id
    # require BOTH cached — download() only short-circuits (no network) when both are present.
    if not ((vdir / "video.mp4").exists() and (vdir / "audio.m4a").exists()):
        return None
    try:
        from ..pipeline.download import download
        from ..pipeline.understand.perceive import perceive
        dl = download(f"https://youtu.be/{video_id}", settings, None)
        return perceive(dl["video_path"], video_id, transcript, sents, settings, None, dl.get("audio_path"))
    except Exception:
        return None


def _round_nan(x):
    return None if x != x else round(x, 3)                  # x!=x → NaN → JSON null


# every Rejection.stage literal the drop ledger can emit — the rejections_* columns count
# per stage from THIS tuple (single source of truth; tests iterate it to catch drift).
# 'build' (W25-G): candidate builders returning None used to die un-ledgered.
REJECTION_STAGES = ("build", "repair", "snap", "dedupe", "post_merge_judge",
                    "post_snap_judge", "quality_floor", "max_clips")


def _integrity_columns(specs, rejections) -> dict:
    """Label-free drop-ledger + judge-integrity columns for one assembled run: per-stage
    rejection counts, merge count, phantom_verdict_rate, ship-flag count/rate, and the
    verified/unverified kill split. Same NaN→None convention as the other columns."""
    m = {f"rejections_{stage}": sum(1 for rj in rejections if rj.stage == stage)
         for stage in REJECTION_STAGES}
    m["n_merged"] = sum(1 for s in specs if s.get("merged"))
    m["phantom_verdict_rate"] = _round_nan(metrics.phantom_verdict_rate(specs, rejections))
    # W25-G: the same rate over QUOTABLE kinds only — absence kinds are unquotable by
    # construction (and by design cannot kill), so this is the actionable gate-health number.
    m["phantom_quotable_rate"] = _round_nan(metrics.phantom_quotable_rate(specs, rejections))
    n_flagged, flagged_rate = metrics.shipped_flagged(specs)
    m["n_shipped_flagged"] = n_flagged
    m["shipped_flagged_rate"] = _round_nan(flagged_rate)
    m["verified_kill"], m["unverified_kill"] = metrics.kill_counts(rejections)
    return m


def _wave2_columns(st, specs, stats) -> dict:
    """Label-free Wave-2 extraction-quality columns for one assembled run (I1a).

    ``stats`` is the machine-readable dict assemble_clips fills (plan_engine,
    n_arcs_detected); everything else derives from the shipped specs + structure.
    plan_engine is surfaced numerically as plan_fallback_rate (plan → 0.0,
    plan-fallback → 1.0); on the priority selector the plan engine never ran, so
    plan_fallback_rate and n_arcs_detected are None (NaN convention) — never a fake 0
    that would dilute the aggregate. Same NaN→None convention as the other columns."""
    units_by_id = st.units_by_id()
    engine = str(stats.get("plan_engine", "") or "")
    m = {
        "chapter_coverage": _round_nan(
            metrics.chapter_coverage(specs, units_by_id, st.content_map.topics())),
        # W25-C sibling: shipped seconds / node seconds — slivers read low, not 1.0
        "topic_span_coverage": _round_nan(
            metrics.topic_span_coverage(specs, st.content_map.topics())),
        "plan_engine": engine,                              # string column (not aggregated)
        "plan_fallback_rate": _round_nan(
            {"plan": 0.0, "plan-fallback": 1.0}.get(engine, float("nan"))),
        "n_arcs_detected": (int(stats["n_arcs_detected"])
                            if "n_arcs_detected" in stats else None),
        "n_arc_clips_shipped": metrics.n_arc_clips(specs),
        "n_trims": metrics.trim_moves(specs),
        # Q1 selection/budget signals (assemble_clips stats; None when assembly never ran):
        "anchor_budget": (int(stats["anchor_budget"]) if "anchor_budget" in stats else None),
        "n_refund_rounds": (int(stats["n_refund_rounds"])
                            if "n_refund_rounds" in stats else None),
        "n_refund_clips": (int(stats["n_refund_clips"])
                           if "n_refund_clips" in stats else None),
        # W25-G columns. Stats-borne counts follow the anchor_budget rule: None (NaN) when
        # assembly never ran/filled them — a missing measurement must not dilute the
        # aggregate as a fake 0. Spec/structure-derived counts are real 0s when clean
        # (they are tripwires: forward_requires_edges and index_clamp_events expect 0).
        "n_refund_superset_replaced": (int(stats["n_refund_superset_replaced"])
                                       if "n_refund_superset_replaced" in stats else None),
        "index_clamp_events": (int(stats["index_clamp_events"])
                               if "index_clamp_events" in stats else None),
        "n_min_duration_extensions": metrics.min_duration_extensions(specs),
        "forward_requires_edges": metrics.forward_requires_edges(st),
        # topic-first-clipping engine columns (safe on non-topic runs: stats returns 0.0)
        "window_len": window_len_stats(specs),
        "topic_selectivity": topic_selectivity(stats),
    }
    m["severed_pairs_linked"], m["severed_pairs_merged"] = metrics.severed_pair_counts(specs)
    return m


# ── trustworthy-eval helpers (freeze / average N runs / variance) ─────────────
def parse_eval_args(argv: list[str]) -> dict:
    """Parse the eval CLI. Positional args are video ids; the rest are flags. Robust to a
    missing/invalid --runs / --topic value (falls back rather than crashing)."""
    out = {"video_ids": [], "topic": None, "verbose": False,
           "runs": 1, "freeze": False, "freeze_specs": False, "rebuild": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--verbose":
            out["verbose"] = True
        elif a == "--freeze":
            out["freeze"] = True
        elif a == "--freeze-specs":
            out["freeze_specs"] = True
        elif a == "--rebuild":
            out["rebuild"] = True
        elif a == "--topic":
            if i + 1 < len(argv):
                out["topic"] = argv[i + 1]
                i += 1
        elif a == "--runs":
            if i + 1 < len(argv):
                try:
                    out["runs"] = max(1, int(argv[i + 1]))
                    i += 1                          # consume the value only if it parsed…
                except ValueError:
                    pass                            # …else leave the token for normal handling
        elif a.startswith("--"):
            pass                                            # unknown flag: ignore
        else:
            out["video_ids"].append(a)
        i += 1
    return out


def resolve_structure(video_id: str, build_fn: Callable[[], object], *,
                      freeze: bool, rebuild: bool, sentences=None) -> tuple[object, str]:
    """Decide whether to reuse the cached Structure or (re)build it.

    freeze + cache present (and not rebuild) → load it, holding structure constant across
    an A/B. --freeze is the explicit hold-anyway override (W25-A): a cache that is STALE
    against the live ``sentences`` still loads (load_structure warns loudly on stderr) and
    is reported as 'cached-stale' so the eval column can't pass a poisoned run off as
    clean. Otherwise build via build_fn and persist it (so the next --freeze run has a
    cache to reuse). Returns (structure, source) with source in
    {"cached","cached-stale","built","rebuilt"}."""
    if freeze and not rebuild:
        cached = load_structure(video_id, sentences, allow_stale=True)
        if cached is not None:
            stale = sentences is not None and structure_is_stale(cached, sentences)
            return cached, ("cached-stale" if stale else "cached")
    st = build_fn()
    save_structure(st)
    return st, ("rebuilt" if rebuild else "built")


def _is_num(x) -> bool:
    """True for a real numeric metric value — excludes None, NaN, bools, and strings."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (x != x)


def mean(values: list) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _metric_values(rows: list[dict], metric: str) -> list[float]:
    """Numeric values of `metric` across a list of per-run metric dicts (drops None/NaN/missing)."""
    return [r[metric] for r in rows if _is_num(r.get(metric))]


def summarize(values: list) -> Optional[dict]:
    """{mean, std, min, max, n} over numeric values (NaN/None dropped). Sample std (n-1) for
    n≥2, else 0.0 — the std is the run-to-run noise a comprehension delta must beat. None if empty."""
    nums = [v for v in values if _is_num(v)]
    if not nums:
        return None
    return {
        "mean": sum(nums) / len(nums),
        "std": statistics.stdev(nums) if len(nums) >= 2 else 0.0,
        "min": min(nums),
        "max": max(nums),
        "n": len(nums),
    }


def select_videos(explicit: list[str], discovered: list[str], cap: int = 8) -> list[str]:
    """Explicitly-listed video ids are honored in full; only the auto-discovered default set is
    capped (so `run_eval v1 … v10` never silently drops requested videos from the headline)."""
    return explicit if explicit else discovered[:cap]


def aggregate_over_runs(results: list[dict], metric: str) -> Optional[dict]:
    """Run-to-run summary of `metric`'s across-video mean, over a FIXED video set so the reported
    std reflects noise — not video-set composition drift. The fixed set is the videos with a
    numeric value at every comparable run index (R = the common/minimum run count across videos);
    a video whose metric flips numeric↔None or has fewer runs is excluded wholesale rather than
    shifting the video set per index. Returns {mean,std,min,max,n,videos} or None."""
    run_lens = [len(res["runs"]) for res in results if res.get("runs")]
    if not run_lens:
        return None
    R = min(run_lens)
    fixed = [res for res in results
             if all(_is_num(res["runs"][r].get(metric)) for r in range(R))]
    if not fixed:
        return None
    per_run_means = [mean([res["runs"][r][metric] for res in fixed]) for r in range(R)]
    s = summarize(per_run_means)
    return {**s, "videos": len(fixed)} if s is not None else None


# ── measurement (fixed structure + specs → metrics dict) ──────────────────────
def _measure(st, specs, sents, adapter, det, topic, gold, settings, verbose=False) -> dict:
    """Compute the label-free (+ gold, when present) metrics for one already-assembled run."""
    units_by_id = st.units_by_id()
    thr = float(settings.get("min_comprehension_score", config.JUDGE_MIN_SCORE))
    mean_score, comp_rate, n_judged, n_err = metrics.comprehension(specs, sents, adapter, topic, thr)
    m = {
        "domain": det.domain,
        "topic": topic or "(none)",
        "n_clips": len(specs),
        "has_perception": st.has_perception,
        "comprehension_rate": round(comp_rate, 3),
        "mean_judge_score": round(mean_score, 3),
        "judge_error_rate": round(n_err / (n_judged + n_err), 3) if (n_judged + n_err) else 0.0,
        "ends_on_period_rate": round(metrics.ends_on_period_rate(specs, sents), 3),
        "unresolved_reference_rate": round(metrics.unresolved_reference_rate(specs, units_by_id), 3),
        "context_complete_rate": round(metrics.context_complete_rate(specs, units_by_id, adapter), 3),
        "prerequisite_gap_rate": round(metrics.prerequisite_gap_rate(specs, units_by_id), 3),
        "worked_example_completeness": _round_nan(metrics.worked_example_completeness(specs, units_by_id, adapter)),
        "visual_completeness": _round_nan(metrics.visual_completeness(specs, units_by_id)),
        "sequence_coherence": round(metrics.sequence_coherence(specs, units_by_id), 3),
        "carded_rate": round(metrics.grounding_ok_rate(specs), 3),
        "grounding_precision": _round_nan(metrics.grounding_precision(specs, units_by_id)),
    }
    m["opening_onset_rate"] = _round_nan(metrics.opening_onset_rate(specs, sents))
    # VID2 edge-probe advisory rates (NaN → null when no probe ran, i.e. the default).
    _edge_starts, _edge_ends = metrics.edge_clean_rates(specs)
    m["edge_starts_clean_rate"] = _round_nan(_edge_starts)
    m["edge_ends_clean_rate"] = _round_nan(_edge_ends)
    if gold.get("units"):
        m["role_accuracy"] = _round_nan(metrics.role_accuracy(st.units, gold["units"]))
    if gold.get("anchors"):
        m["anchor_recall"] = _round_nan(metrics.anchor_recall(gold["anchors"], specs))
        m["boundary_error"] = _round_nan(metrics.boundary_error(gold["anchors"], specs))
    gold_ch = gold_chapters(gold)
    if gold_ch:
        # Segmentation gold from creator YouTube chapters (YTSeg protocol): reference labels
        # by greatest time overlap; window k from the TRUE (reference) segmentation; hypothesis
        # boundaries at both content-map granularities. NaN (→ null) when a granularity is absent.
        ref = metrics.assign_segments(sents, gold_ch)
        k = metrics.window_size(ref)
        for gran, nodes in (("topics", st.content_map.topics()),
                            ("chapters", st.content_map.chapters())):
            hyp = metrics.assign_segments(sents, [{"start": n.start, "end": n.end} for n in nodes])
            m[f"pk_{gran}"] = _round_nan(metrics.pk(ref, hyp, k))
            m[f"windowdiff_{gran}"] = _round_nan(metrics.windowdiff(ref, hyp, k))
        m["clip_straddle_rate"] = _round_nan(metrics.clip_straddle_rate(specs, gold_ch))
    if gold.get("inventory"):
        # W25-G standing inventory recall (gold-gated): the coverage-audit ≥0.60 one-sided
        # rule over the golden 'inventory' items — qP finally counts in gold aggregates.
        cov = metrics.inventory_coverage(gold["inventory"], specs)
        m["inventory_recall"] = _round_nan(metrics.inventory_recall(gold["inventory"], specs))
        if verbose:
            print(f"    inventory coverage (one-sided overlap ≥ {metrics.INVENTORY_COVER_FRAC}):")
            for it, frac, ok in cov:
                tag = "covered" if ok else "MISSED "
                print(f"      [{tag}] item {it.get('n', '?'):>2} "
                      f"({str(it.get('type', '') or '-'):18}) "
                      f"{float(it.get('start', 0.0)):7.1f}-{float(it.get('end', 0.0)):7.1f}  "
                      f"overlap={frac:.2f}  {str(it.get('description', ''))[:60]}")
    if verbose:
        print(f"    per-clip judge (role, score, failures):")
        for role, score, fails, err in metrics.judge_failures(specs, sents, adapter, topic):
            tag = " UNJUDGED" if err else ""
            print(f"      [{role or '-':20}] score={score}  fails={fails}{tag}")
    return m


def eval_video(video_id: str, topic_override: str | None, verbose: bool = False, *,
               runs: int = 1, freeze: bool = False, freeze_specs: bool = False,
               rebuild: bool = False) -> dict | None:
    """Assemble + score a video `runs` times. What is held constant per run:
        default        rebuild structure + reassemble + judge   → total variance
        --freeze       structure loaded once; reassemble + judge → assembly + judge variance
        --freeze-specs structure + specs fixed; only re-judge    → judge-only variance
    Returns {video_id, domain, topic, source, n_runs, runs:[metricdict,…]} or None."""
    tpath = config.WORK_DIR / video_id / "transcript.json"
    if not tpath.exists():
        print(f"  [skip] {video_id}: no cached transcript")
        return None
    transcript = json.loads(tpath.read_text())
    transcript.setdefault("title", "")
    sents = _sentences(transcript, video_id)
    gold = load_golden(video_id) or {}
    topic = topic_override or (gold.get("topics") or [""])[0]

    settings = dict(config.DEFAULTS)
    adapter, det = select_adapter(transcript, settings)
    # perception is topic-independent + slow — compute once and reuse across runs so no-freeze
    # runs vary only the understanding-stage LLM calls (the meaningful structure noise).
    perception = _maybe_perceive(video_id, transcript, sents, settings)

    def build_fn():
        return build_structure(video_id, transcript, sents, adapter, det, settings, None, perception)

    hold_structure = freeze or freeze_specs
    url = f"https://youtu.be/{video_id}"
    run_dicts, source = [], None
    st = specs = None
    stats: dict = {}
    for r in range(runs):
        # whole iteration is guarded: a transient build/assembly/judge failure loses only THIS
        # run (structure/specs stay as-is and are retried next iteration) instead of discarding
        # every already-collected run for the video.
        try:
            if st is None or not hold_structure:
                st, source = resolve_structure(video_id, build_fn, freeze=hold_structure,
                                               rebuild=rebuild, sentences=sents)
            if specs is None or not freeze_specs:
                stats = {}                     # per-run stats (held with specs under freeze-specs)
                specs, _notes, rejections = assemble_clips(st, topic, sents, url, video_id,
                                                           settings, adapter, stats=stats)
                # W25-G: every assembled run leaves an auditable plan/arcs/shipped/ledger
                # trail (the qP diagnosis needed offline reconstruction — never again).
                write_run_artifacts(video_id, specs, rejections, stats)
            m = _measure(st, specs, sents, adapter, det, topic, gold, settings,
                         verbose=verbose and r == 0)
            # W25-G string column (not aggregated, like plan_engine): where THIS run's
            # structure came from — 'cached-stale' flags a freeze-override on a poisoned cache.
            m["structure_source"] = source or ""
            m.update(_integrity_columns(specs, rejections))
            m.update(_wave2_columns(st, specs, stats))
            run_dicts.append(m)
        except Exception as e:  # noqa: BLE001
            print(f"    [run {r + 1}/{runs} error] {e}")
            continue
    if not run_dicts:
        return None
    return {"video_id": video_id, "domain": det.domain, "topic": topic or "(none)",
            "source": source or "built", "n_runs": len(run_dicts), "runs": run_dicts}


# ── reporting ─────────────────────────────────────────────────────────────────
_PREFERRED_KEYS = ["comprehension_rate", "mean_judge_score", "ends_on_period_rate",
                   "unresolved_reference_rate", "context_complete_rate", "prerequisite_gap_rate",
                   "worked_example_completeness", "visual_completeness", "sequence_coherence",
                   "carded_rate", "grounding_precision",
                   "edge_starts_clean_rate", "edge_ends_clean_rate",
                   "role_accuracy", "anchor_recall",
                   "boundary_error", "pk_topics", "windowdiff_topics", "pk_chapters",
                   "windowdiff_chapters", "clip_straddle_rate", "inventory_recall",
                   "chapter_coverage", "topic_span_coverage", "plan_fallback_rate",
                   "n_arcs_detected",
                   "n_arc_clips_shipped", "n_trims", "severed_pairs_linked",
                   "severed_pairs_merged", "anchor_budget", "n_refund_rounds",
                   "n_refund_clips", "n_refund_superset_replaced",
                   "n_min_duration_extensions", "index_clamp_events",
                   "forward_requires_edges", "phantom_verdict_rate",
                   "phantom_quotable_rate",
                   "shipped_flagged_rate", "n_shipped_flagged", "verified_kill",
                   "unverified_kill", "n_clips"]


def _numeric_keys(results: list[dict]) -> list[str]:
    """All keys that are numeric in at least one run, preferred order first then any extras."""
    seen = set()
    for res in results:
        for run in res["runs"]:
            for k, v in run.items():
                if _is_num(v):
                    seen.add(k)
    ordered = [k for k in _PREFERRED_KEYS if k in seen]
    ordered += sorted(seen - set(ordered))
    return ordered


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _print_video_row(res: dict) -> None:
    runs = res["runs"]
    tag = f" [{res['source']}]" if res.get("source") else ""
    if len(runs) == 1:
        print(f"  {json.dumps(runs[0])}{tag}")
        return
    print(f"  source={res['source']} runs={res['n_runs']}")
    for k in _numeric_keys([res]):
        s = summarize([run.get(k) for run in runs])
        if s:
            print(f"    {k:30} {_fmt(s['mean'])} ± {_fmt(s['std'])}  "
                  f"[{_fmt(s['min'])}, {_fmt(s['max'])}]")


def _print_aggregate(results: list[dict]) -> None:
    """Headline: run-to-run variance of each metric's across-video mean, over a fixed video set.
    The comprehension_rate std is the noise floor a real improvement has to clear."""
    print("\n=== AGGREGATE (across-video mean per run; mean ± run-to-run std, fixed video set) ===")
    for k in _numeric_keys(results):
        s = aggregate_over_runs(results, k)
        if s:
            print(f"  {k:30} {_fmt(s['mean'])} ± {_fmt(s['std'])}  "
                  f"[{_fmt(s['min'])}, {_fmt(s['max'])}]  (runs={s['n']}, videos={s['videos']})")


def main(argv: list[str]) -> None:
    a = parse_eval_args(argv)
    discovered = sorted(
        Path(p).parent.name for p in glob.glob(str(config.WORK_DIR / "*/transcript.json")))
    vids = select_videos(a["video_ids"], discovered, cap=8)
    if not a["video_ids"] and len(discovered) > len(vids):
        print(f"(showing first {len(vids)} of {len(discovered)} cached videos; list ids explicitly for more)")
    mode = ("freeze-specs" if a["freeze_specs"] else "freeze" if a["freeze"]
            else "rebuild" if a["rebuild"] else "full-rebuild")
    print(f"eval: {len(vids)} video(s), runs={a['runs']}, mode={mode}")

    results = []
    for vid in vids:
        print(f"→ {vid} …")
        try:
            res = eval_video(vid, a["topic"], verbose=a["verbose"], runs=a["runs"],
                             freeze=a["freeze"], freeze_specs=a["freeze_specs"], rebuild=a["rebuild"])
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {vid}: {e}")
            res = None
        if res:
            results.append(res)
            _print_video_row(res)

    if results:
        _print_aggregate(results)


if __name__ == "__main__":
    main(sys.argv[1:])
