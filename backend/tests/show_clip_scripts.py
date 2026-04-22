"""Print the Whisper transcript of every reel in an audit CSV.

For each row, we already know [t_start, t_end]. Slice the cached full-video
WAV on that window, feed it to faster-whisper once, and print the resulting
text next to the clip's metadata. Parallelises across 4 workers since
CTranslate2 releases the GIL during inference.

Run from the worktree root:
    ./backend/.venv/bin/python backend/tests/show_clip_scripts.py \
        --csv audit_after_PR7i.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.services.clip_whisper_refine import _FULL_VIDEO_AUDIO_CACHE  # noqa: E402


def _seed_cache_from_tmpdir() -> int:
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    seeded = 0
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


_MODEL = None
_MODEL_LOCK = threading.Lock()


def _model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from faster_whisper import WhisperModel
                _MODEL = WhisperModel("small.en", device="cpu", compute_type="int8")
    return _MODEL


def _transcribe_clip(video_id: str, t_start: float, t_end: float) -> str:
    from faster_whisper.audio import decode_audio

    wav = _FULL_VIDEO_AUDIO_CACHE.get(video_id)
    if wav is None or not wav.exists():
        return f"(no cached audio for {video_id})"
    audio = decode_audio(str(wav), sampling_rate=16000)
    sr = 16000
    s = max(0, int(t_start * sr))
    e = min(len(audio), int(t_end * sr))
    if e <= s:
        return "(empty window)"
    sub = audio[s:e]
    segments, _ = _model().transcribe(
        sub,
        language="en",
        word_timestamps=False,
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    seeded = _seed_cache_from_tmpdir()
    print(f"[seeded {seeded} cached WAVs]  [{len(rows)} clips to transcribe]\n")

    # Warm the model on the main thread so all worker calls hit a ready engine.
    _model()

    def _one(idx_row):
        idx, r = idx_row
        try:
            text = _transcribe_clip(r["video_id"], float(r["t_start"]), float(r["t_end"]))
        except Exception as exc:
            text = f"(whisper failed: {exc})"
        return idx, text

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for idx, text in pool.map(_one, list(enumerate(rows))):
            results[idx] = text

    # Print in CSV order so the user can correlate with the audit summary.
    current_vid = None
    for idx, r in enumerate(rows):
        vid = r["video_id"]
        if vid != current_vid:
            print("\n" + "▰" * 80)
            print(f"VIDEO {vid}  ·  {r['content_type']}  ·  query: \"{r['target_query']}\"")
            print("▰" * 80)
            current_vid = vid

        mid_w = r.get("starts_mid_word") == "True"
        mid_s = r.get("ends_mid_sentence") == "True"
        filler = r.get("starts_on_filler") == "True"
        usable = r.get("is_usable") == "True"
        flags = []
        if mid_w: flags.append("MID-WORD-START")
        if mid_s: flags.append("MID-SENT-END")
        if filler: flags.append("FILLER-START")
        flag_str = " ⚠️  " + " / ".join(flags) if flags else ""
        verdict = "✅ usable" if usable else "❌ flagged"

        label = r.get("label", "").strip()
        label_str = f"  label={label!r}" if label else ""

        print(
            f"\n  ── clip {r['clip_index']}  "
            f"[{float(r['t_start']):.2f}s → {float(r['t_end']):.2f}s · "
            f"{float(r['duration_sec']):.1f}s]  {verdict}{flag_str}{label_str}"
        )
        print(f"  {results[idx]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
