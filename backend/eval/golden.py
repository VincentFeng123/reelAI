"""Golden-set loading + time alignment (spec §14).

A golden file (``eval/golden/<video_id>.json``) carries human labels:
    {
      "video_id", "url", "domain", "content_type",
      "topics": ["projectile motion"],
      "reference_concepts": ["kinematics", ...],
      "units":   [{"start","end","role","concepts_introduced","concepts_required","is_anchor"}],
      "anchors": [{"anchor_role","start","end","required_elements_present","prerequisites",
                   "must_understand_without_source"}],
      "chapters": [{"start_time","end_time","title"}],   # creator YouTube chapters
      "chapters_provenance": "creator",                  # written by make_golden --chapters
      "human": {"clips": [{"start","end","understandable","failure_kinds",
                           "needed_first","labeled_at"}],
                "video_note": "…"}                       # written by POST /api/labels (labeling page)
    }
All fields are optional except video_id — label-free metrics work with none of them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# human labels ↔ judge/manifest clip spans join when BOTH endpoints agree within this (s)
HUMAN_MATCH_TOL_S = 0.5


def load_golden(video_id: str) -> Optional[dict]:
    p = GOLDEN_DIR / f"{video_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def gold_chapters(gold: Optional[dict]) -> list[dict]:
    """Creator-chapter segmentation gold from a loaded golden file. Tolerant of a missing
    gold dict and of files without chapters (returns []). Normalizes yt-dlp's
    {start_time,end_time,title} (also accepts {start,end}) to float {start,end,title},
    drops malformed or empty spans, and returns the list time-sorted."""
    if not gold:
        return []
    out = []
    for ch in gold.get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        try:
            # explicit None must fall back to the other key form too (dict.get's default
            # only covers a MISSING key), so {"start_time": null, "start": 3} still parses.
            s_raw = ch.get("start_time")
            e_raw = ch.get("end_time")
            s = float(ch.get("start") if s_raw is None else s_raw)
            e = float(ch.get("end") if e_raw is None else e_raw)
        except (TypeError, ValueError):
            continue
        if e > s:
            out.append({"start": s, "end": e, "title": str(ch.get("title") or "")})
    return sorted(out, key=lambda c: (c["start"], c["end"]))


# ── human labels ('human' block: judge-calibration ground truth, spec E1) ─────
def human_block(gold: Optional[dict]) -> dict:
    """The 'human' labels block of a loaded golden file, tolerant of a missing gold dict,
    a missing key, and malformed shapes: always {"clips": [dict, ...], "video_note": str}."""
    h = (gold or {}).get("human")
    if not isinstance(h, dict):
        return {"clips": [], "video_note": ""}
    clips = [c for c in (h.get("clips") or []) if isinstance(c, dict)]
    return {"clips": clips, "video_note": str(h.get("video_note") or "")}


def spans_match(a_start, a_end, b_start, b_end, tol: float = HUMAN_MATCH_TOL_S) -> bool:
    """The human-label join rule: True when BOTH endpoints agree within `tol` seconds.
    Malformed values never match (a broken label must not swallow a real one)."""
    try:
        return (abs(float(a_start) - float(b_start)) <= tol
                and abs(float(a_end) - float(b_end)) <= tol)
    except (TypeError, ValueError):
        return False


def find_human_clip(clips, start, end, tol: float = HUMAN_MATCH_TOL_S) -> Optional[dict]:
    """First human clip label whose span matches (start, end) within `tol` — the join used
    by both the labels endpoint (upsert) and judge_calibration (human↔judge pairing)."""
    for c in clips or []:
        if isinstance(c, dict) and spans_match(c.get("start"), c.get("end"), start, end, tol):
            return c
    return None


def _norm_human_label(label: dict, labeled_at: str) -> Optional[dict]:
    """Normalize one incoming label to the golden 'human.clips' schema. None (skipped) when
    start/end are unparseable or Q1 was never answered — an unanswered row must never
    overwrite an answered label."""
    try:
        start, end = float(label["start"]), float(label["end"])
    except (KeyError, TypeError, ValueError):
        return None
    understandable = label.get("understandable")
    if not isinstance(understandable, bool):
        return None
    return {"start": round(start, 2), "end": round(end, 2), "understandable": understandable,
            "failure_kinds": [str(k) for k in (label.get("failure_kinds") or [])],
            "needed_first": str(label.get("needed_first") or ""),
            "labeled_at": str(label.get("labeled_at") or labeled_at)}


def merge_human_labels(gold: Optional[dict], video_id: str, labels: list, video_note: str = "",
                       labeled_at: str = "") -> dict:
    """Pure merge of human labels into a golden dict — NEVER clobbers other gold keys (same
    contract as make_golden.merge_chapters_into_golden; video_id set only when absent).
    Clips upsert by span match within HUMAN_MATCH_TOL_S — relabeling a clip replaces its
    previous label, new spans append — and stay time-sorted (deterministic files). The
    video_note replaces only when the incoming note is non-empty (a save with an untouched
    note field must not erase an earlier one). Returns the updated gold dict."""
    data = dict(gold or {})
    data.setdefault("video_id", video_id)
    existing = human_block(data)
    clips = [dict(c) for c in existing["clips"]]
    for raw in labels or []:
        lab = _norm_human_label(raw if isinstance(raw, dict) else {}, labeled_at)
        if lab is None:
            continue
        clips = [c for c in clips
                 if not spans_match(c.get("start"), c.get("end"), lab["start"], lab["end"])]
        clips.append(lab)
    clips.sort(key=lambda c: (float(c.get("start", 0.0)), float(c.get("end", 0.0))))
    note = str(video_note or "").strip() or existing["video_note"]
    data["human"] = {"clips": clips, "video_note": note}
    return data


def merge_human_into_golden(video_id: str, labels: list, video_note: str = "",
                            labeled_at: str = "", golden_dir: Optional[Path] = None) -> dict:
    """Merge a labeling-page save into <golden_dir>/<video_id>.json and return the saved
    'human' block. Mirrors make_golden.merge_chapters_into_golden's safety contract: an
    existing file that is unreadable or not a JSON object raises ValueError instead of
    being clobbered — a hand-authored golden file must never lose its labels."""
    gdir = Path(golden_dir) if golden_dir is not None else GOLDEN_DIR
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / f"{video_id}.json"
    existing: dict = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"unreadable golden file {path}, not merging: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"golden file {path} is not a JSON object, not merging")
        existing = raw
    data = merge_human_labels(existing, video_id, labels, video_note, labeled_at)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data["human"]


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def best_match(start: float, end: float, items: list[dict], thresh: float = 0.5) -> Optional[dict]:
    best, best_iou = None, thresh
    for it in items:
        s = iou(start, end, float(it.get("start", 0)), float(it.get("end", 0)))
        if s >= best_iou:
            best, best_iou = it, s
    return best
