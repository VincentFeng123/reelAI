"""Dump a golden *skeleton* from the predicted structure, for a human to correct (spec §4.2).

Running the full understanding pass and emitting the predicted units in the golden schema makes
hand-labeling fast: a human fixes roles / concept lists / anchor spans instead of typing them from
scratch. Output goes to ``eval/golden/<video_id>.skeleton.json`` and NEVER overwrites a
hand-authored ``<video_id>.json`` — rename the skeleton to that once corrected.

A second, independent mode imports creator YouTube chapters as segmentation gold (YTSeg,
EACL 2024: creator chapters are accepted segmentation ground truth). It fetches metadata only
(yt_dlp ``extract_info(download=False)``, no media download) and MERGES
``{"chapters": [...], "chapters_provenance": "creator"}`` into ``eval/golden/<video_id>.json``
without clobbering any hand-authored keys. Videos without creator chapters are reported and
nothing is written.

Usage:
    python -m backend.eval.make_golden <video_id> [--topic "the derivative"]
    python -m backend.eval.make_golden --chapters <video_id> [<video_id> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import config
from ..adapters import select_adapter
from ..pipeline.sentences import build_sentence_index, sentences_from_chunks
from ..pipeline.understand import build_structure
from .golden import GOLDEN_DIR


def _sentences(transcript):
    s = build_sentence_index(transcript)
    if transcript.get("source") == "supadata":
        frac = (sum(1 for x in s if x.ends_with_period) / len(s)) if s else 0.0
        avg = (sum(x.end - x.start for x in s) / len(s)) if s else 999.0
        if len(s) < 5 or frac < 0.3 or avg > 40.0:
            s = sentences_from_chunks(transcript.get("chunks", []))
    return s


def make_skeleton(video_id: str, topic: str | None = None) -> dict:
    tpath = config.WORK_DIR / video_id / "transcript.json"
    if not tpath.exists():
        raise SystemExit(f"no cached transcript for {video_id} at {tpath}")
    transcript = json.loads(tpath.read_text(encoding="utf-8"))
    transcript.setdefault("title", "")
    sents = _sentences(transcript)
    settings = dict(config.DEFAULTS)
    adapter, det = select_adapter(transcript, settings)
    st = build_structure(video_id, transcript, sents, adapter, det, settings, None)

    ref_concepts = sorted({c for u in st.units for c in u.concepts_introduced})[:40]
    return {
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}",
        "domain": det.domain,
        "content_type": det.content_type,
        "topics": [topic] if topic else [],
        "reference_concepts": ref_concepts,
        "units": [{
            "start": round(u.start, 2), "end": round(u.end, 2), "role": u.role,
            "concepts_introduced": u.concepts_introduced, "concepts_required": u.concepts_required,
            "is_anchor": adapter.is_anchor_role(u.role),
        } for u in st.units],
        "anchors": [{
            "anchor_role": u.role, "start": round(u.start, 2), "end": round(u.end, 2),
            "required_elements_present": adapter.required_elements(u.role),
            "prerequisites": u.concepts_required, "must_understand_without_source": True,
        } for u in st.units if adapter.is_anchor_role(u.role)],
        "_note": "SKELETON of PREDICTED structure — correct roles/concepts/anchors by hand, then "
                 "rename to <video_id>.json to use as ground truth.",
    }


# ── creator-chapter segmentation gold (--chapters mode) ──────────────────────
def fetch_chapters(video_id: str) -> list[dict]:
    """Creator chapters via the yt_dlp Python API — metadata only
    (``extract_info(download=False)``), never a media download. yt_dlp is already a pipeline
    dependency (see backend/pipeline/download.py). Returns normalized
    [{start_time, end_time, title}], or [] when the video has no creator chapters."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://youtu.be/{video_id}", download=False)
    out = []
    for ch in (info or {}).get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        try:
            s, e = float(ch.get("start_time")), float(ch.get("end_time"))
        except (TypeError, ValueError):
            continue
        out.append({"start_time": round(s, 2), "end_time": round(e, 2),
                    "title": str(ch.get("title") or "")})
    return out


def merge_chapters_into_golden(video_id: str, chapters: list[dict],
                               golden_dir: Path = GOLDEN_DIR) -> Path:
    """MERGE {"chapters", "chapters_provenance": "creator"} into <golden_dir>/<video_id>.json.
    Every pre-existing key (hand-labeled units/anchors/topics/…) is preserved — only the two
    chapter keys are (re)written; video_id is set only when absent. An existing file that is
    unreadable or not a JSON object raises ValueError instead of being clobbered — a
    hand-authored golden file with a stray syntax error must never lose its labels."""
    golden_dir.mkdir(parents=True, exist_ok=True)
    path = golden_dir / f"{video_id}.json"
    data: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"unreadable golden file {path}, not merging: {e}") from e
        if not isinstance(existing, dict):
            raise ValueError(f"golden file {path} is not a JSON object, not merging")
        data = existing
    data.setdefault("video_id", video_id)
    data["chapters"] = chapters
    data["chapters_provenance"] = "creator"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def run_chapters_mode(video_ids: list[str], golden_dir: Path = GOLDEN_DIR) -> None:
    """Import creator chapters for each id; report + write nothing when a video has none."""
    for vid in video_ids:
        try:
            chapters = fetch_chapters(vid)
        except Exception as e:  # noqa: BLE001 — one bad video must not kill the batch
            print(f"{vid}: chapter fetch failed: {e}")
            continue
        if not chapters:
            print(f"{vid}: no creator chapters")
            continue
        try:
            path = merge_chapters_into_golden(vid, chapters, golden_dir)
        except ValueError as e:      # corrupt golden file: skip this video, keep the batch going
            print(f"{vid}: {e}")
            continue
        print(f"{vid}: merged {len(chapters)} creator chapters -> {path}")


def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    if "--chapters" in argv:
        if not args:
            raise SystemExit(
                "usage: python -m backend.eval.make_golden --chapters <video_id> [<video_id> ...]")
        run_chapters_mode(args)
        return
    if not args:
        raise SystemExit('usage: python -m backend.eval.make_golden <video_id> [--topic "..."]')
    topic = None
    if "--topic" in argv:
        i = argv.index("--topic")
        topic = argv[i + 1] if i + 1 < len(argv) else None
    video_id = args[0]
    skeleton = make_skeleton(video_id, topic)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / f"{video_id}.skeleton.json"
    out.write_text(json.dumps(skeleton, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({len(skeleton['units'])} units, {len(skeleton['anchors'])} anchors, "
          f"domain={skeleton['domain']})")


if __name__ == "__main__":
    main(sys.argv[1:])
