#!/usr/bin/env python3
"""
Mass clip audit harness — runs the clip-cutting pipeline across a pinned
corpus of ~25 videos covering 6 content types, then measures each emitted
clip's boundary quality using Whisper word-level timestamps as the
ground-truth oracle.

Binary metrics (oracle-robust; trust these):
    starts_mid_word    - first phoneme is cut
    ends_mid_sentence  - last word lacks terminal punctuation
    starts_on_filler   - first word is 'um' / 'uh' / 'like' / ...

Continuous metrics (±50-100 ms whisper noise floor; report deltas only):
    start_precision_ms - distance from t_start to first-word onset
    end_precision_ms   - distance from t_end   to last-word offset

Headline KPI (purely mechanical):
    usable_clip_rate = NOT starts_mid_word AND NOT ends_mid_sentence
                       AND NOT starts_on_filler
                       AND start_precision_ms < 500
                       AND end_precision_ms   < 500

Run:
    python backend/tests/mass_clip_audit.py \\
        --corpus backend/tests/mass_clip_audit_corpus.yaml \\
        --audit-cache-suffix baseline \\
        --out audit_baseline.csv

Cache / DB isolation:
    The harness forces DATA_DIR=<cwd>/audit_work/<suffix> before any app
    import so the pipeline's SQLite DB + caches live in an isolated
    directory. Every suffix = fresh cache namespace. Production caches
    are untouched.

Stat-sig note (copy into the summary):
    n = 25 videos × 3-5 clips ≈ 75-125 boundary observations total.
    Per-content-type buckets have n ≈ 12-20 — aggregate delta across
    all 25 is the decision signal; per-bucket deltas are directional
    and require a ≥15 pp swing before reading anything into them.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

_backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_backend_root))

# The pipeline invokes `yt-dlp` via subprocess, resolving through PATH. When
# the harness is launched with `./backend/.venv/bin/python` without venv
# activation, the system PATH lacks the venv's bin dir, so yt-dlp silently
# fails with FileNotFoundError and _download_clip_audio returns None. Prepend
# this interpreter's bin directory so subprocess sees the venv's binaries.
_interpreter_bin = str(Path(sys.executable).parent)
os.environ["PATH"] = _interpreter_bin + os.pathsep + os.environ.get("PATH", "")

# Cache / DB isolation — must precede every app import so get_settings()
# reads the overridden DATA_DIR on first call.
def _prime_isolated_data_dir(suffix: str) -> Path:
    root = Path.cwd() / "audit_work" / suffix
    root.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_DIR"] = str(root)
    # Force a clean SQLite cache each run; transcripts must be re-fetched.
    for name in ("studyreels.db", "studyreels.db-wal", "studyreels.db-shm"):
        p = root / name
        if p.exists():
            p.unlink()
    return root

# Parse --audit-cache-suffix eagerly so the data-dir override fires before
# any `from app.*` import binds to a stale Settings() instance.
def _peek_suffix() -> str:
    for i, a in enumerate(sys.argv[1:], 1):
        if a.startswith("--audit-cache-suffix="):
            return a.split("=", 1)[1] or "baseline"
        if a == "--audit-cache-suffix" and i + 1 < len(sys.argv):
            return sys.argv[i + 1] or "baseline"
    return "baseline"

_AUDIT_SUFFIX = _peek_suffix()
_ISOLATED_DATA_DIR = _prime_isolated_data_dir(_AUDIT_SUFFIX)

# .env is for API keys (Groq, Gemini) — we still want those loaded so the
# LLM picker paths run. Only DATA_DIR is overridden.
_env_path = _backend_root / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "DATA_DIR":
            continue
        os.environ.setdefault(k, v)

from app.db import init_db  # noqa: E402
from app.ingestion.models import IngestTranscriptCue, IngestTranscriptWord  # noqa: E402
from app.services.clip_boundary import _FILLER_TOKENS, _word_is_filler  # noqa: E402
from app.services.clip_whisper_refine import (  # noqa: E402
    WhisperWord,
    _call_whisper,
    _download_clip_audio,
)
from app.services.topic_cut import (  # noqa: E402
    TranscriptCue,
    cut_video_into_topic_reels,
)

logger = logging.getLogger("mass_clip_audit")

_TERMINAL_PUNCT = {".", "!", "?", "…"}
_USABLE_PRECISION_CEILING_MS = 500.0


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #


@dataclass
class CorpusEntry:
    video_id: str
    content_type: str
    target_query: str
    note: str = ""


def load_corpus(path: Path) -> list[CorpusEntry]:
    try:
        import yaml  # PyYAML ships with the backend deps
    except ImportError:
        raise SystemExit("PyYAML not installed; pip install pyyaml") from None
    raw = yaml.safe_load(path.read_text())
    out: list[CorpusEntry] = []
    for entry in raw or []:
        vid = str(entry.get("video_id") or "").strip()
        if not vid:
            continue
        out.append(CorpusEntry(
            video_id=vid,
            content_type=str(entry.get("content_type") or "unknown"),
            target_query=str(entry.get("target_query") or "").strip(),
            note=str(entry.get("note") or ""),
        ))
    return out


# --------------------------------------------------------------------------- #
# Transcript fetch — mirror real_search_simulation.py preference order
# --------------------------------------------------------------------------- #


def fetch_transcript(video_id: str) -> tuple[list[dict], str]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return [], ""
    try:
        api = YouTubeTranscriptApi()
        tlist = api.list(video_id)
        try:
            tr = tlist.find_manually_created_transcript(["en"])
            kind = "manual"
        except Exception:
            try:
                tr = tlist.find_generated_transcript(["en"])
                kind = "generated"
            except Exception:
                tr = next(iter(tlist), None)
                kind = "other"
        if tr is None:
            return [], ""
        fetched = tr.fetch()
        return (
            [
                {"text": s.text, "start": float(s.start), "duration": float(s.duration)}
                for s in fetched.snippets
            ],
            kind,
        )
    except Exception as exc:
        logger.warning("transcript fetch failed for %s: %s", video_id, exc)
        return [], ""


def build_transcript_cues(raw: list[dict]) -> list[TranscriptCue]:
    out: list[TranscriptCue] = []
    for entry in raw:
        text = str(entry.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        out.append(TranscriptCue(
            start=float(entry["start"]),
            duration=float(entry.get("duration") or 0.0),
            text=text,
        ))
    return out


# --------------------------------------------------------------------------- #
# Per-clip Whisper ground-truth measurement
# --------------------------------------------------------------------------- #


@dataclass
class ClipMetrics:
    video_id: str
    content_type: str
    target_query: str
    transcript_kind: str
    clip_index: int
    t_start: float
    t_end: float
    duration_sec: float
    label: str
    boundary_quality: str
    # Whisper-ground-truth
    whisper_first_word: str = ""
    whisper_last_word: str = ""
    start_precision_ms: float = -1.0
    end_precision_ms: float = -1.0
    starts_mid_word: bool = False
    ends_mid_sentence: bool = False
    starts_on_filler: bool = False
    is_usable: bool = False
    error: str = ""


def measure_clip_with_whisper(
    video_id: str, t_start: float, t_end: float,
) -> dict[str, Any]:
    dl_start = max(0.0, float(t_start) - 1.0)
    dl_end = float(t_end) + 1.0
    tmpdir = Path(tempfile.mkdtemp(prefix="mass_audit_"))
    try:
        audio = _download_clip_audio(video_id, dl_start, dl_end, tmpdir)
        if audio is None:
            return {"error": "download_failed"}
        words = _call_whisper(audio, dl_start)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not words:
        return {"error": "no_whisper_words"}

    # starts_mid_word — does any Whisper word's span straddle t_start
    # by more than 50 ms on each side? 50 ms is tighter than the Whisper
    # intrinsic noise floor, so a true mid-word cut stands out.
    starts_mid_word = any(
        w.start < t_start - 0.05 and w.end > t_start + 0.05
        for w in words
    )

    in_window = [
        w for w in words if (t_start - 0.1) <= w.start and w.end <= (t_end + 0.1)
    ]
    if not in_window:
        return {"error": "no_words_in_window", "all_words": len(words)}

    first_w = in_window[0]
    last_w = in_window[-1]
    start_precision_ms = abs(float(t_start) - first_w.start) * 1000.0
    end_precision_ms = abs(float(t_end) - last_w.end) * 1000.0

    last_text = (last_w.text or "").strip()
    ends_mid_sentence = not (last_text and last_text[-1] in _TERMINAL_PUNCT)
    starts_on_filler = _word_is_filler(first_w.text or "")
    is_usable = (
        not starts_mid_word
        and not ends_mid_sentence
        and not starts_on_filler
        and start_precision_ms < _USABLE_PRECISION_CEILING_MS
        and end_precision_ms < _USABLE_PRECISION_CEILING_MS
    )
    return {
        "first_word": first_w.text,
        "last_word": last_w.text,
        "start_precision_ms": start_precision_ms,
        "end_precision_ms": end_precision_ms,
        "starts_mid_word": starts_mid_word,
        "ends_mid_sentence": ends_mid_sentence,
        "starts_on_filler": starts_on_filler,
        "is_usable": is_usable,
    }


# --------------------------------------------------------------------------- #
# Per-video run
# --------------------------------------------------------------------------- #


def run_entry(entry: CorpusEntry, *, use_llm: bool = True) -> list[ClipMetrics]:
    if entry.video_id.startswith("TODO-"):
        logger.info("skipping unresolved TODO entry: %s", entry.video_id)
        return []

    raw, kind = fetch_transcript(entry.video_id)
    if not raw:
        logger.warning("no transcript for %s — skipped", entry.video_id)
        return []

    duration_sec = float(raw[-1]["start"]) + float(raw[-1].get("duration") or 0.0)
    tc_cues = build_transcript_cues(raw)

    try:
        classification, topic_reels = cut_video_into_topic_reels(
            entry.video_id,
            query=entry.target_query,
            duration_sec=duration_sec,
            use_llm=use_llm,
            refine_boundaries=True,
            transcript=tc_cues,
            info_dict=None,
            ingest_cues_for_precision=None,
            silence_ranges=None,
            user_min_sec=20.0,
            user_max_sec=55.0,
            user_target_sec=55.0,
        )
    except Exception:
        logger.exception("cut_video_into_topic_reels raised for %s", entry.video_id)
        return []

    if classification.is_short:
        logger.info("%s classified as short; no clip work", entry.video_id)
        return []
    if not topic_reels:
        logger.info("%s produced zero clips", entry.video_id)
        return []

    rows: list[ClipMetrics] = []
    for idx, r in enumerate(topic_reels, 1):
        m = ClipMetrics(
            video_id=entry.video_id,
            content_type=entry.content_type,
            target_query=entry.target_query,
            transcript_kind=kind,
            clip_index=idx,
            t_start=float(r.t_start),
            t_end=float(r.t_end),
            duration_sec=float(r.duration_sec),
            label=str(getattr(r, "label", "") or ""),
            boundary_quality=str(getattr(r, "boundary_quality", "") or ""),
        )
        probe = measure_clip_with_whisper(entry.video_id, r.t_start, r.t_end)
        if "error" in probe:
            m.error = str(probe["error"])
            rows.append(m)
            continue
        m.whisper_first_word = str(probe["first_word"])
        m.whisper_last_word = str(probe["last_word"])
        m.start_precision_ms = float(probe["start_precision_ms"])
        m.end_precision_ms = float(probe["end_precision_ms"])
        m.starts_mid_word = bool(probe["starts_mid_word"])
        m.ends_mid_sentence = bool(probe["ends_mid_sentence"])
        m.starts_on_filler = bool(probe["starts_on_filler"])
        m.is_usable = bool(probe["is_usable"])
        rows.append(m)
    return rows


# --------------------------------------------------------------------------- #
# CSV + summary writers
# --------------------------------------------------------------------------- #


_CSV_FIELDS = [
    "video_id", "content_type", "target_query", "transcript_kind",
    "clip_index", "t_start", "t_end", "duration_sec",
    "label", "boundary_quality",
    "whisper_first_word", "whisper_last_word",
    "start_precision_ms", "end_precision_ms",
    "starts_mid_word", "ends_mid_sentence", "starts_on_filler",
    "is_usable", "error",
]


def write_csv(rows: list[ClipMetrics], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: asdict(r)[k] for k in _CSV_FIELDS})


def _bucket_rate(rows: list[ClipMetrics], predicate) -> tuple[int, int, float]:
    total = sum(1 for r in rows if not r.error)
    hits = sum(1 for r in rows if not r.error and predicate(r))
    rate = hits / total if total else 0.0
    return hits, total, rate


def _median_or_none(vals: list[float]) -> float | None:
    vals = [v for v in vals if v >= 0]
    return statistics.median(vals) if vals else None


def write_summary(rows: list[ClipMetrics], out: Path, suffix: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    ap = lines.append
    total = len(rows)
    clean = [r for r in rows if not r.error]
    ap(f"# Mass clip audit — suffix={suffix}")
    ap("")
    ap(f"- total rows: {total}")
    ap(f"- clean rows (whisper measured): {len(clean)}")
    ap(f"- error rows: {total - len(clean)}")
    ap("")
    ap("**Statistical-significance note.** Aggregate delta across all 25 videos "
       "is the decision signal; per-content-type deltas are directional and "
       "require a ≥15 pp swing before reading anything into a single bucket.")
    ap("")
    ap("**Continuous metrics caveat.** Whisper measures Whisper-refined "
       "boundaries — the noise floor is ~±50-100 ms. Do not publish absolute "
       "precision numbers; compare only against the prior run's CSV.")
    ap("")

    # Overall binary rollup
    smw_h, smw_t, smw_r = _bucket_rate(rows, lambda r: r.starts_mid_word)
    ems_h, ems_t, ems_r = _bucket_rate(rows, lambda r: r.ends_mid_sentence)
    sof_h, sof_t, sof_r = _bucket_rate(rows, lambda r: r.starts_on_filler)
    use_h, use_t, use_r = _bucket_rate(rows, lambda r: r.is_usable)
    ap("## Aggregate binary metrics")
    ap("")
    ap(f"| metric | hits | total | rate |")
    ap(f"|---|---|---|---|")
    ap(f"| starts_mid_word | {smw_h} | {smw_t} | {smw_r:.3f} |")
    ap(f"| ends_mid_sentence | {ems_h} | {ems_t} | {ems_r:.3f} |")
    ap(f"| starts_on_filler | {sof_h} | {sof_t} | {sof_r:.3f} |")
    ap(f"| **usable_clip_rate** | {use_h} | {use_t} | **{use_r:.3f}** |")
    ap("")

    start_med = _median_or_none([r.start_precision_ms for r in clean])
    end_med = _median_or_none([r.end_precision_ms for r in clean])
    ap("## Aggregate continuous metrics (compare across runs; noisy in absolute)")
    ap("")
    ap(f"- start_precision_ms median: {start_med:.1f}" if start_med is not None else "- start_precision_ms: n/a")
    ap(f"- end_precision_ms median: {end_med:.1f}" if end_med is not None else "- end_precision_ms: n/a")
    ap("")

    # Per-content-type rollup
    by_type: dict[str, list[ClipMetrics]] = {}
    for r in rows:
        by_type.setdefault(r.content_type, []).append(r)
    ap("## Per-content-type rollup (directional; ≥15pp swing required)")
    ap("")
    ap("| type | n | mid_word | mid_sent | filler | usable |")
    ap("|---|---|---|---|---|---|")
    for ct in sorted(by_type):
        bucket = by_type[ct]
        n = sum(1 for r in bucket if not r.error)
        if not n:
            ap(f"| {ct} | 0 | - | - | - | - |")
            continue
        mw = sum(1 for r in bucket if not r.error and r.starts_mid_word) / n
        ms = sum(1 for r in bucket if not r.error and r.ends_mid_sentence) / n
        fl = sum(1 for r in bucket if not r.error and r.starts_on_filler) / n
        us = sum(1 for r in bucket if not r.error and r.is_usable) / n
        ap(f"| {ct} | {n} | {mw:.2f} | {ms:.2f} | {fl:.2f} | {us:.2f} |")
    ap("")

    # Error breakdown
    errors: dict[str, int] = {}
    for r in rows:
        if r.error:
            errors[r.error] = errors.get(r.error, 0) + 1
    if errors:
        ap("## Error breakdown")
        ap("")
        for k, v in sorted(errors.items(), key=lambda kv: -kv[1]):
            ap(f"- {k}: {v}")
        ap("")

    out.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--audit-cache-suffix", default="baseline")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=0,
                        help="0 = all; >0 = first N for smoke tests")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip LLM segmentation; use heuristic/semantic path only")
    parser.add_argument("--skip-ids", default="",
                        help="comma-separated video_ids to skip (e.g. stuck long videos)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Surface the "dropping out-of-range segment" diagnostics from the
    # pipeline so we can see exactly which LLM picks die and why.
    logging.getLogger("app.services.topic_cut").setLevel(logging.DEBUG)
    logging.getLogger("app.services.clip_boundary").setLevel(logging.DEBUG)

    logger.info("data_dir (isolated): %s", _ISOLATED_DATA_DIR)
    init_db()

    corpus = load_corpus(args.corpus)
    if args.max_videos > 0:
        corpus = corpus[: args.max_videos]
    skip_ids = {s.strip() for s in args.skip_ids.split(",") if s.strip()}
    logger.info("loaded %d corpus entries (skip=%s)", len(corpus), sorted(skip_ids) or "-")

    all_rows: list[ClipMetrics] = []
    summary_path = args.out.with_name(args.out.stem + "_summary.md")
    t0 = time.time()
    for i, entry in enumerate(corpus, 1):
        if entry.video_id in skip_ids:
            logger.info("[%d/%d] SKIP %s (via --skip-ids)", i, len(corpus), entry.video_id)
            continue
        logger.info(
            "[%d/%d] %s type=%s query=%r",
            i, len(corpus), entry.video_id, entry.content_type, entry.target_query,
        )
        try:
            rows = run_entry(entry, use_llm=not args.no_llm)
        except Exception:
            logger.exception("entry failed: %s", entry.video_id)
            continue
        all_rows.extend(rows)
        logger.info("  → %d clip(s)", len(rows))
        # Flush after every video so a kill mid-run preserves completed work.
        write_csv(all_rows, args.out)
        write_summary(all_rows, summary_path, args.audit_cache_suffix)

    elapsed = time.time() - t0
    logger.info("done in %.0fs — wrote %s and %s", elapsed, args.out, summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
