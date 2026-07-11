"""Judge-calibration math over the human labels (Hamel protocol). Pure math — no LLM.

Usage:
    python -m backend.eval.judge_calibration [--manifest PATH] [--golden-dir PATH]
                                             [--resamples N] [--seed N]

Joins the golden files' human blocks (written by the labeling page via POST /api/labels)
to the labeling manifest's judge fields — matched by video_id + start/end BOTH within
golden.HUMAN_MATCH_TOL_S (0.5 s) — and reports:

  * Cohen's kappa between human Q1 ("could someone who has not seen the video follow
    this?") and the judge's `understandable`, with a 1000-resample percentile-bootstrap
    CI. Interpretation anchors (design doc item 9): the expert ceiling on this criterion
    is kappa ~= 0.51, so kappa ~0.5 vs the judge IS success — do not chase 0.8. Decision
    rule: kappa < 0.4 replace the judge; >= 0.6 trust and tune.
  * Per-failure-kind precision/recall of judge KILLS against the human failure-kind
    checklist. Any kind with fewer than 10 human positives REFUSES to print a stat
    ("n=X insufficient") — below that the estimate is noise (design doc item 10).
  * A recommended action per kind: trust (precision >= 0.7 on >= 10 human labels),
    distrust, or need-more-labels.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from .. import config
from .golden import GOLDEN_DIR, HUMAN_MATCH_TOL_S, find_human_clip, human_block

DEFAULT_MANIFEST = config.STATIC_DIR / "labeling" / "manifest.json"
MIN_HUMAN_POSITIVES = 10      # below this, per-kind precision/recall is refused, not printed
TRUST_PRECISION = 0.70        # kill authority only above this precision (design doc item 10)
DEFAULT_RESAMPLES = 1000
EXPERT_CEILING_KAPPA = 0.51


# ── Cohen's kappa + bootstrap (pure) ──────────────────────────────────────────
def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Cohen's kappa over (human, judge) boolean pairs. NaN when there are no pairs or
    chance agreement is 1 (both raters constant — kappa undefined, 0/0)."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    a = sum(1 for h, j in pairs if h and j)
    b = sum(1 for h, j in pairs if h and not j)
    c = sum(1 for h, j in pairs if not h and j)
    d = n - a - b - c
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def bootstrap_kappa_ci(pairs: list[tuple[bool, bool]], n_resamples: int = DEFAULT_RESAMPLES,
                       seed: int = 0, alpha: float = 0.05) -> Optional[tuple[float, float, int]]:
    """Percentile-bootstrap CI on Cohen's kappa: resample the joined pairs with replacement
    n_resamples times, take the alpha/2 and 1-alpha/2 percentiles. Deterministic given
    `seed`. Returns (lo, hi, n_valid) — degenerate resamples whose kappa is undefined are
    dropped (n_valid counts the rest) — or None when pairs is empty / nothing is defined."""
    if not pairs:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    ks: list[float] = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        k = cohens_kappa(sample)
        if k == k:                              # drop NaN (degenerate resample)
            ks.append(k)
    if not ks:
        return None
    ks.sort()
    lo_i = int((alpha / 2.0) * len(ks))
    hi_i = min(len(ks) - 1, int((1.0 - alpha / 2.0) * len(ks)))
    return ks[lo_i], ks[hi_i], len(ks)


# ── join human labels ↔ manifest judge fields (pure) ─────────────────────────
def join_labels(entries: list[dict], human_blocks: dict[str, dict],
                tol: float = HUMAN_MATCH_TOL_S) -> list[dict]:
    """One joined row per manifest entry with a matching answered human label — matched by
    video_id + start/end both within `tol` (golden.find_human_clip). Unanswered labels
    (understandable not a bool) never join."""
    joined: list[dict] = []
    for e in entries or []:
        block = human_blocks.get(str(e.get("video_id", ""))) or {}
        lab = find_human_clip(block.get("clips") or [], e.get("start"), e.get("end"), tol)
        if lab is None or not isinstance(lab.get("understandable"), bool):
            continue
        judge = e.get("judge") or {}
        joined.append({
            "video_id": str(e.get("video_id", "")),
            "start": e.get("start"), "end": e.get("end"),
            "status": str(e.get("status", "")),
            "stratum": str(e.get("stratum", "")),
            "human_understandable": bool(lab["understandable"]),
            "human_kinds": [str(k) for k in (lab.get("failure_kinds") or [])],
            "judge_understandable": bool(judge.get("understandable", False)),
            "judge_score": judge.get("score"),
            "judge_kinds": [str(k) for k in (judge.get("failure_kinds") or [])],
        })
    return joined


def kappa_pairs(joined: list[dict]) -> list[tuple[bool, bool]]:
    return [(r["human_understandable"], r["judge_understandable"]) for r in joined]


# ── per-failure-kind precision/recall of judge kills (pure) ───────────────────
def recommended_action(stat: dict) -> str:
    """trust / distrust / need-more-labels per the design-doc rule: kill authority only for
    kinds with precision >= 0.7 on >= 10 human labels. A kind with enough human positives
    but ZERO judge kills has no precision to trust or distrust → need-more-labels."""
    if not stat.get("sufficient"):
        return "need-more-labels"
    p = stat.get("precision")
    if p is None:
        return "need-more-labels"
    return "trust" if p >= TRUST_PRECISION else "distrust"


def per_kind_stats(joined: list[dict], min_positives: int = MIN_HUMAN_POSITIVES) -> dict:
    """Per failure kind (union of kinds on judge kills and human labels): precision/recall
    of judge KILLS carrying that kind vs the human checklist. A kind with fewer than
    `min_positives` human positives gets NO precision/recall keys at all — only
    {n_human, n_judge_kills, sufficient: False, action: need-more-labels} — printing a
    stat off <10 positives would be noise, so it is refused, not caveated."""
    kills = [r for r in joined if r.get("status") == "rejected"]
    kinds = sorted({k for r in kills for k in r.get("judge_kinds", [])}
                   | {k for r in joined for k in r.get("human_kinds", [])})
    out: dict[str, dict] = {}
    for kind in kinds:
        n_human = sum(1 for r in joined if kind in r.get("human_kinds", []))
        kill_k = [r for r in kills if kind in r.get("judge_kinds", [])]
        stat: dict = {"n_human": n_human, "n_judge_kills": len(kill_k),
                      "sufficient": n_human >= min_positives}
        if stat["sufficient"]:
            tp = sum(1 for r in kill_k if kind in r.get("human_kinds", []))
            stat["precision"] = (tp / len(kill_k)) if kill_k else None
            stat["recall"] = tp / n_human
        stat["action"] = recommended_action(stat)
        out[kind] = stat
    return out


# ── I/O + reporting ───────────────────────────────────────────────────────────
def load_human_blocks(golden_dir: Optional[Path] = None) -> dict[str, dict]:
    """{video_id: human block} for every golden file carrying human labels. Unreadable
    files and files without a human block are skipped (never fatal)."""
    gdir = Path(golden_dir) if golden_dir is not None else GOLDEN_DIR
    out: dict[str, dict] = {}
    for p in sorted(gdir.glob("*.json")):
        try:
            gold = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(gold, dict):
            continue
        h = human_block(gold)
        if h["clips"] or h["video_note"]:
            out[str(gold.get("video_id") or p.stem)] = h
    return out


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def print_report(joined: list[dict], n_resamples: int = DEFAULT_RESAMPLES, seed: int = 0) -> None:
    pairs = kappa_pairs(joined)
    print(f"joined human↔judge pairs: {len(pairs)} "
          f"(kills: {sum(1 for r in joined if r['status'] == 'rejected')})")
    if not pairs:
        print("no joined pairs — label some clips first (labeling page → POST /api/labels)")
        return
    k = cohens_kappa(pairs)
    ci = bootstrap_kappa_ci(pairs, n_resamples=n_resamples, seed=seed)
    ci_txt = f"95% bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}] over {ci[2]} resamples" \
        if ci else "CI unavailable"
    print(f"\nCohen's kappa (human Q1 vs judge understandable): {_fmt(k if k == k else None)}  "
          f"({ci_txt})")
    print(f"  expert ceiling on this criterion ≈ {EXPERT_CEILING_KAPPA}: kappa ~0.5 IS success. "
          "Decision rule: <0.4 replace judge; >=0.6 trust and tune.")
    stats = per_kind_stats(joined)
    if not stats:
        print("\nno failure kinds recorded on either side yet")
        return
    print("\nper-failure-kind: judge kills vs human checklist "
          f"(stats refused below {MIN_HUMAN_POSITIVES} human positives)")
    for kind, st in stats.items():
        if not st["sufficient"]:
            print(f"  {kind:28} n={st['n_human']} insufficient "
                  f"(<{MIN_HUMAN_POSITIVES} human positives) → need-more-labels")
            continue
        print(f"  {kind:28} n_human={st['n_human']:>3} kills={st['n_judge_kills']:>3} "
              f"precision={_fmt(st['precision'])} recall={_fmt(st['recall'])} → {st['action']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m backend.eval.judge_calibration",
        description="Judge-vs-human calibration: Cohen's kappa + bootstrap CI and "
                    "per-failure-kind kill precision/recall. Pure math, no LLM.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, metavar="PATH",
                   help=f"labeling manifest (default {DEFAULT_MANIFEST})")
    p.add_argument("--golden-dir", type=Path, default=None, metavar="DIR",
                   help="golden directory holding the human blocks (default eval/golden/)")
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES, metavar="N",
                   help="bootstrap resamples (default 1000)")
    p.add_argument("--seed", type=int, default=0, metavar="N",
                   help="bootstrap RNG seed (deterministic CI)")
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    a = parse_args(argv)
    if not a.manifest.exists():
        print(f"no manifest at {a.manifest} — run "
              "`python -m backend.eval.sample_for_labeling --collect <video_id …>` first")
        return
    try:
        payload = json.loads(a.manifest.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"unreadable manifest {a.manifest}: {e}")
        return
    entries = payload.get("entries") or []
    human = load_human_blocks(a.golden_dir)
    print(f"manifest entries: {len(entries)}; videos with human labels: {len(human)}")
    joined = join_labels(entries, human)
    print_report(joined, n_resamples=a.resamples, seed=a.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
