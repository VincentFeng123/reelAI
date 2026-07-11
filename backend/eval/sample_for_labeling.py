"""Export a labeling manifest for human calibration of the clip-only judge (Hamel protocol).

Usage:
    python -m backend.eval.sample_for_labeling --collect <video_id> [...] [--out PATH] [--limit N]

Runs the SAME cached-structure drive path as backend/eval/run_eval.py / judge_probe.py
(``assemble_clips`` over ``work/<id>/structure.json`` + cached transcript — never the
network; reuses judge_probe.load_probe_inputs, including its cached-detection adapter
resolution) and writes ``backend/static/labeling/manifest.json`` with one entry per
ACCEPTED spec AND per judge-stage REJECTION (repair / post_merge_judge / post_snap_judge).
Accepted specs are (re)judged exactly like the eval comprehension metric
(``visual_summary=""``, the spec's own context card) so the manifest carries the same judge
fields the headline number is computed from. Invoked for real this makes LIVE judge calls —
fine at the CLI, NEVER in tests: every manifest-building function below is pure and takes
the drive-path products (specs / rejections / sentences / a judge_fn) as inputs.

Bias hygiene: each entry's judge fields live under a separate "judge" key and the labeling
page (backend/static/labeling/index.html) keeps them collapsed behind a details toggle the
labeler is told to open only AFTER answering — never shown alongside the embed by default.

Strata (the per-entry "stratum" tag; --limit caps EACH stratum, deterministically):
    kill:<kind>   judge-stage rejection, keyed by its first confirmed failure kind. The
                  corruption probe says reasoning_complete/result_complete kills are the
                  least-validated part of the judge — kills get labeled first.
    band_4_7      accepted spec whose judge score fell in [0.4, 0.7] — the maximum-
                  disagreement band the calibration sweep needs most.
    random        every other accepted spec (the unbiased core for kappa/PPI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from .. import config
from ..embed import embed_url
from ..pipeline.assemble import assemble_clips
from .judge_probe import load_probe_inputs

DEFAULT_OUT = config.STATIC_DIR / "labeling" / "manifest.json"

# validate.FailureReason's kind vocabulary (keep in sync with backend/pipeline/assemble/validate.py)
JUDGE_KIND_VOCAB = ("unresolved_reference", "missing_prerequisite", "missing_visual",
                    "missing_problem_statement", "missing_reasoning", "missing_result",
                    "not_source_grounded", "off_topic", "other")
# human-only checklist additions (boundary failures the judge has no kind for)
EXTRA_HUMAN_KINDS = ("starts_mid_thought", "ends_unresolved", "boundary_garbage")
HUMAN_KIND_OPTIONS = JUDGE_KIND_VOCAB + EXTRA_HUMAN_KINDS

# the Rejection stages that are judge verdicts (mechanical drops are not judge decisions)
JUDGE_REJECTION_STAGES = ("repair", "post_merge_judge", "post_snap_judge")

BAND_LO, BAND_HI = 0.4, 0.7                    # judge-score band for the 'band_4_7' stratum


# ── pure manifest builders (no LLM, no I/O — tests exercise these directly) ──
# embed_url is imported from backend.embed (the shared canonical helper) — re-exported here
# so sample_for_labeling.embed_url (and its tests) keep resolving.
def span_text(spec: dict, sentences) -> str:
    """Full transcript text of an accepted spec's sentence span (mirrors metrics._clip_text)."""
    i0 = int(spec.get("sentence_start_idx", 0))
    i1 = int(spec.get("sentence_end_idx", i0))
    i0 = max(0, min(i0, len(sentences) - 1))
    i1 = max(i0, min(i1, len(sentences) - 1))
    return " ".join((sentences[i].text or "") for i in range(i0, i1 + 1)).strip()


def time_span_text(start: float, end: float, sentences) -> str:
    """Transcript text of every sentence overlapping [start, end] — Rejections carry times,
    not sentence indices, so the span is recovered by time overlap."""
    out = [(s.text or "") for s in sentences
           if min(float(s.end), float(end)) - max(float(s.start), float(start)) > 0.0]
    return " ".join(out).strip()


def spec_status(spec: dict) -> str:
    """'shipped_flagged' when the asymmetric gate shipped it past unverifiable judge
    concerns (Wave 1), else 'shipped'."""
    flagged = bool(spec.get("ship_flagged")) or \
        "unverified_judge_concerns" in (spec.get("warnings") or ())
    return "shipped_flagged" if flagged else "shipped"


def stratum_for(status: str, score, failure_kinds) -> str:
    """'kill:<kind>' for judge-stage rejections (first kind; 'other' when none recorded),
    'band_4_7' for accepted specs whose judge score sits in [0.4, 0.7], else 'random'."""
    if status == "rejected":
        kinds = [str(k) for k in (failure_kinds or []) if k]
        return f"kill:{kinds[0] if kinds else 'other'}"
    if score is not None and BAND_LO <= float(score) <= BAND_HI:
        return "band_4_7"
    return "random"


def _entry(video_id: str, title: str, start: float, end: float, text: str, card: str,
           status: str, judge: dict) -> dict:
    return {
        "id": f"{video_id}:{start:.2f}-{end:.2f}:{status}",
        "video_id": video_id,
        "video_title": title,
        "start": round(float(start), 2),
        "end": round(float(end), 2),
        "embed_url": embed_url(video_id, start, end),
        "text": text,
        "context_card": card,
        "status": status,                     # shipped | shipped_flagged | rejected
        "stratum": stratum_for(status, judge.get("score"), judge.get("failure_kinds")),
        # bias hygiene: the labeling page renders this ONLY inside a collapsed details
        # toggle the labeler opens after answering — never next to the embed.
        "judge": judge,
    }


def build_entries(video_id: str, title: str, specs: list[dict], rejections, sentences,
                  judge_fn: Callable[[str, str, str], dict]) -> list[dict]:
    """One manifest entry per accepted spec (judge fields from judge_fn — LIVE at the CLI,
    a stub in tests) and per judge-stage rejection (judge fields from the Rejection record;
    no extra judge call — the kill verdict already exists). Deterministic ordering:
    (start, end, status, stratum)."""
    entries: list[dict] = []
    for s in specs or []:
        text = span_text(s, sentences)
        # governing contract (contract_role, P1) — mirrors metrics.comprehension's brief;
        # anchor-role fallback keeps older cached specs identical.
        judge = dict(judge_fn(text, s.get("contract_role") or s.get("role", "") or "",
                              s.get("context_card", "") or ""))
        judge.setdefault("stage", "accept")
        judge.setdefault("reason", None)
        entries.append(_entry(video_id, title, float(s["start"]), float(s["end"]), text,
                              s.get("context_card", "") or "", spec_status(s), judge))
    for r in rejections or []:
        if getattr(r, "stage", "") not in JUDGE_REJECTION_STAGES:
            continue                          # mechanical drops are not judge verdicts
        judge = {"score": (None if r.score is None else round(float(r.score), 3)),
                 "understandable": False,
                 "failure_kinds": [str(k) for k in (r.failure_kinds or [])],
                 "error": False, "stage": r.stage, "reason": r.reason}
        entries.append(_entry(video_id, title, float(r.start), float(r.end),
                              time_span_text(float(r.start), float(r.end), sentences),
                              "", "rejected", judge))
    entries.sort(key=lambda e: (e["start"], e["end"], e["status"], e["stratum"]))
    seen: dict[str, int] = {}                 # ids must be unique (UI radio groups key on them)
    for e in entries:
        n = seen.get(e["id"], 0)
        seen[e["id"]] = n + 1
        if n:
            e["id"] = f"{e['id']}#{n}"
    return entries


def apply_stratum_limit(entries: list[dict], limit: Optional[int]) -> list[dict]:
    """Keep at most `limit` entries PER STRATUM, preserving the (already deterministic)
    input order — so --limit N is reproducible run to run. None → everything."""
    if limit is None:
        return list(entries)
    seen: dict[str, int] = {}
    out: list[dict] = []
    for e in entries:
        k = e.get("stratum", "")
        if seen.get(k, 0) >= limit:
            continue
        seen[k] = seen.get(k, 0) + 1
        out.append(e)
    return out


def build_manifest(per_video: list[tuple[str, str, list[dict]]], limit: Optional[int] = None,
                   stamp: str = "") -> dict:
    """Final manifest payload: videos in the order collected, entries per-video-sorted,
    per-stratum cap applied globally. `stamp` is supplied by the caller (CLI uses datetime;
    tests pass a fixed string) so the builders stay deterministic."""
    videos = [{"video_id": vid, "title": title} for vid, title, _ents in per_video]
    entries: list[dict] = []
    for _vid, _title, ents in per_video:
        entries.extend(ents)
    entries = apply_stratum_limit(entries, limit)
    strata: dict[str, int] = {}
    for e in entries:
        strata[e["stratum"]] = strata.get(e["stratum"], 0) + 1
    return {"version": 1, "generated_at": stamp,
            "kind_options": list(HUMAN_KIND_OPTIONS),
            "videos": videos, "n_entries": len(entries),
            "strata": dict(sorted(strata.items())), "entries": entries}


def write_manifest(payload: dict, out_path: Path = DEFAULT_OUT) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# ── drive path (LIVE judge calls at the CLI; tests inject judge_fn/products) ──
def video_title(video_id: str) -> str:
    """work/<id>/title.txt when present, else the cached transcript's title, else ''."""
    vdir = config.WORK_DIR / video_id
    tpath = vdir / "title.txt"
    if tpath.exists():
        try:
            return tpath.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            return ""
    trpath = vdir / "transcript.json"
    if trpath.exists():
        try:
            return str(json.loads(trpath.read_text(encoding="utf-8")).get("title") or "")
        except Exception:  # noqa: BLE001
            return ""
    return ""


def make_live_judge_fn(adapter, topic: str) -> Callable[[str, str, str], dict]:
    """LIVE judge fields for an accepted spec — mirrors metrics.comprehension's judge call
    exactly (visual_summary="", the spec's own context card, temperature 0.0 and model
    routing inside judge_clip). CLI-only; tests never call this."""
    from ..pipeline.assemble.validate import judge_clip

    def judge_fn(text: str, role: str, context_card: str) -> dict:
        v = judge_clip(text, role, adapter, visual_summary="", topic=topic,
                       context_card=context_card)
        return {"score": (None if v.error else round(float(v.score), 3)),
                "understandable": bool(v.understandable),
                "failure_kinds": [f.kind for f in v.failure_reasons],
                "error": bool(v.error)}
    return judge_fn


def collect_video(video_id: str,
                  judge_fn: Optional[Callable[[str, str, str], dict]] = None
                  ) -> Optional[tuple[str, list[dict]]]:
    """Run the cached-structure drive path for one video and build its manifest entries.
    Returns (title, entries), or None when the transcript/structure cache is missing
    (load_probe_inputs prints the skip note). judge_fn defaults to the LIVE judge."""
    ctx = load_probe_inputs(video_id)
    if ctx is None:
        return None
    url = f"https://youtu.be/{video_id}"
    specs, _notes, rejections = assemble_clips(ctx["structure"], ctx["topic"], ctx["sents"],
                                               url, video_id, ctx["settings"], ctx["adapter"])
    if judge_fn is None:
        judge_fn = make_live_judge_fn(ctx["adapter"], ctx["topic"])
    title = video_title(video_id)
    return title, build_entries(video_id, title, specs, rejections, ctx["sents"], judge_fn)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m backend.eval.sample_for_labeling",
        description="Export the human-labeling manifest (accepted specs + judge-stage "
                    "rejections) over cached structures. LIVE judge calls on accepted specs.")
    p.add_argument("--collect", nargs="+", required=True, metavar="video_id",
                   help="video id(s) with cached work/<id>/transcript.json + structure.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, metavar="PATH",
                   help=f"manifest output path (default {DEFAULT_OUT})")
    p.add_argument("--limit", type=_positive_int, default=None, metavar="N",
                   help="cap entries PER STRATUM (deterministic: first N in span order)")
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    a = parse_args(argv)
    print(f"sample_for_labeling: {len(a.collect)} video(s) — cached-structure drive path, "
          "LIVE judge calls on accepted specs")
    per_video: list[tuple[str, str, list[dict]]] = []
    for vid in a.collect:
        print(f"→ {vid} …")
        try:
            res = collect_video(vid)
        except Exception as e:  # noqa: BLE001 — one bad video must not sink the export
            print(f"  [error] {vid}: {e}")
            res = None
        if res is None:
            continue
        title, entries = res
        n_kills = sum(1 for e in entries if e["status"] == "rejected")
        print(f"  {len(entries)} entr(y/ies) ({n_kills} judge-stage rejection(s))")
        per_video.append((vid, title, entries))
    if not per_video:
        print("nothing to write — no requested video had cached inputs + entries")
        return
    from datetime import datetime                # CLI-only; builders take an explicit stamp
    payload = build_manifest(per_video, limit=a.limit,
                             stamp=datetime.now().isoformat(timespec="seconds"))
    path = write_manifest(payload, a.out)
    print(f"wrote {path}  ({payload['n_entries']} entries; strata {payload['strata']})")
    print("label at: http://localhost:8000/labeling/index.html")


if __name__ == "__main__":
    main(sys.argv[1:])
