"""Visualise clip boundaries with before/after word context.

For every row in audit_after_PR7i.csv, transcribe a ~5 s window centred on
t_start and a ~5 s window centred on t_end using faster-whisper with
word-level timestamps, then print the words split into BEFORE / AFTER the
boundary with a clear marker. Lets a human eyeball whether each clip
actually begins at a sentence-start and ends at a sentence-end.

Run from the worktree root:
    ./backend/.venv/bin/python backend/tests/show_boundary_context.py \
        --csv audit_after_PR7i.csv --limit 15
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the app package importable when this file is run directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.services.clip_whisper_refine import (  # noqa: E402
    _FULL_VIDEO_AUDIO_CACHE,
    _ensure_full_video_audio,
)


CONTEXT_BEFORE_SEC = 4.0
CONTEXT_AFTER_SEC = 4.0


def _seed_cache_from_tmpdir() -> int:
    """Scan $TMPDIR for prior-run reelai-ingest-clip-audio-* dirs and seed
    `_FULL_VIDEO_AUDIO_CACHE` with any {video_id}.wav we find, newest first.
    Saves re-downloading full videos when this script is run back-to-back with
    the audit harness."""
    import os
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    seeded = 0
    # Sort dirs newest→oldest so the freshest file wins.
    dirs = sorted(
        tmp.glob("reelai-ingest-clip-audio-*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for d in dirs:
        if not d.is_dir():
            continue
        for wav in d.glob("*.wav"):
            vid = wav.stem
            if vid in _FULL_VIDEO_AUDIO_CACHE:
                continue
            if wav.stat().st_size == 0:
                continue
            _FULL_VIDEO_AUDIO_CACHE[vid] = wav
            seeded += 1
    return seeded


def _transcribe_window(audio_path: Path, t0: float, t1: float) -> list[tuple[float, float, str]]:
    """Return [(word_start, word_end, text), ...] inside [t0, t1]."""
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    model = _cached_model()
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    # Slice audio at sample level to keep Whisper context tight.
    t0 = max(0.0, t0)
    t1 = max(t0 + 0.1, t1)
    sr = 16000
    start_sample = int(t0 * sr)
    end_sample = min(len(audio), int(t1 * sr))
    sub = audio[start_sample:end_sample]
    segments, _ = model.transcribe(
        sub,
        language="en",
        word_timestamps=True,
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    words: list[tuple[float, float, str]] = []
    for seg in segments:
        for w in seg.words or []:
            abs_start = float(w.start) + t0
            abs_end = float(w.end) + t0
            words.append((abs_start, abs_end, w.word))
    return words


_MODEL_SINGLETON = None


def _cached_model():
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        from faster_whisper import WhisperModel
        _MODEL_SINGLETON = WhisperModel("small", device="cpu", compute_type="int8")
    return _MODEL_SINGLETON


def _fmt_words_split(
    words: list[tuple[float, float, str]],
    boundary: float,
    side: str,
) -> str:
    """side='start' → BEFORE clipped out, AFTER clipped in.
    side='end'   → BEFORE clipped in, AFTER clipped out."""
    before = [w for w in words if w[1] <= boundary + 0.01]
    after = [w for w in words if w[0] >= boundary - 0.01]
    # Words that straddle the boundary (rare w/ word_timestamps but possible).
    straddle = [w for w in words if w[0] < boundary and w[1] > boundary]
    if side == "start":
        before_label = "[BEFORE clip — cut away]"
        after_label = "[AFTER boundary — clip begins]"
    else:
        before_label = "[BEFORE boundary — clip ends]"
        after_label = "[AFTER clip — cut away]"

    def _render(ws: list[tuple[float, float, str]]) -> str:
        return " ".join(w[2].strip() for w in ws).strip() or "(no words)"

    parts = [
        f"    {before_label}:",
        f"        {_render(before)}",
    ]
    if straddle:
        sw = straddle[0]
        parts.append(
            f"    [STRADDLES boundary @ {boundary:.3f}s]:  "
            f"{sw[2].strip()!r} ({sw[0]:.3f}s → {sw[1]:.3f}s)"
        )
    parts += [
        f"    ──── boundary @ {boundary:.3f}s ────",
        f"    {after_label}:",
        f"        {_render(after)}",
    ]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="audit CSV path")
    ap.add_argument("--limit", type=int, default=15, help="clips to render")
    ap.add_argument(
        "--filter",
        choices=["all", "usable", "not_usable"],
        default="all",
        help="show only usable / not-usable clips (default: all)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"csv not found: {csv_path}")
        return 1

    rows: list[dict] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if args.filter == "usable" and r.get("is_usable") != "True":
                continue
            if args.filter == "not_usable" and r.get("is_usable") != "False":
                continue
            rows.append(r)

    rows = rows[: args.limit]
    seeded = _seed_cache_from_tmpdir()
    print(f"Seeded {seeded} cached WAVs from $TMPDIR")
    print(f"Rendering {len(rows)} clip boundaries (context ±{CONTEXT_BEFORE_SEC:.1f}s)\n")

    for i, r in enumerate(rows, 1):
        vid = r["video_id"]
        idx = r["clip_index"]
        t_start = float(r["t_start"])
        t_end = float(r["t_end"])
        label = r.get("label", "")
        q = r.get("boundary_quality", "")
        is_usable = r.get("is_usable", "")
        mid_word = r.get("starts_mid_word", "")
        mid_sent = r.get("ends_mid_sentence", "")
        sp = r.get("start_precision_ms", "")
        ep = r.get("end_precision_ms", "")

        print("═" * 80)
        print(
            f"[{i:2d}/{len(rows)}] {vid} · clip {idx} · {r.get('content_type','?')} · "
            f"\"{r.get('target_query','')}\"  label={label!r}  q={q}"
        )
        print(
            f"    t_start={t_start:.3f}s  t_end={t_end:.3f}s  "
            f"dur={float(r['duration_sec']):.2f}s  usable={is_usable}  "
            f"mid_word={mid_word}  mid_sent={mid_sent}  "
            f"start_prec={sp}ms  end_prec={ep}ms"
        )

        audio = _ensure_full_video_audio(vid)
        if audio is None:
            print("    (failed to fetch audio)\n")
            continue

        # START boundary
        t0 = max(0.0, t_start - CONTEXT_BEFORE_SEC)
        t1 = t_start + CONTEXT_AFTER_SEC
        try:
            words = _transcribe_window(audio, t0, t1)
        except Exception as exc:
            print(f"    (whisper error on start window: {exc})")
            words = []
        print("  ── START boundary context ──")
        print(_fmt_words_split(words, t_start, "start"))

        # END boundary
        t0 = max(0.0, t_end - CONTEXT_BEFORE_SEC)
        t1 = t_end + CONTEXT_AFTER_SEC
        try:
            words = _transcribe_window(audio, t0, t1)
        except Exception as exc:
            print(f"    (whisper error on end window: {exc})")
            words = []
        print("  ── END boundary context ──")
        print(_fmt_words_split(words, t_end, "end"))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
