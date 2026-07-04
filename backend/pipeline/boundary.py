"""Precise boundary refinement with targeted Whisper.

Supadata gives fast but coarse (caption-level) timing. After the LLM picks rough
clip ranges, we transcribe ONLY a small audio window around each boundary with
faster-whisper (word-level timestamps + punctuation), then snap:
  - start → the start of the sentence containing the rough start
  - end   → the end of the nearest period-terminated sentence at/after the rough end

This buys word-level precision for a few seconds of audio per boundary instead of
transcribing the whole video.
"""
from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from .. import config
from ..errors import PipelineError
from . import download as download_mod
from .sentences import Sentence, build_sentence_index
from .transcribe import _get_whisper

ProgressCb = Optional[Callable[[float, str], None]]


def _ensure_audio(url: str, video_id: str) -> Path:
    work = config.WORK_DIR / video_id
    work.mkdir(parents=True, exist_ok=True)
    audio = work / "audio.m4a"
    if audio.exists():
        return audio
    video = work / "video.mp4"
    if video.exists():
        download_mod._extract_audio(video, audio)
        return audio
    # audio-only download (fast, not throttled like video)
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work / "audio_src.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": str(Path(config.FFMPEG_BIN).parent),
        "remote_components": ["ejs:github"],
        "retries": 5,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:  # noqa: BLE001
        raise PipelineError(f"Audio download failed: {str(e).strip().splitlines()[-1]}")
    cands = sorted(work.glob("audio_src.*"))
    if not cands:
        raise PipelineError("Audio download produced no file.")
    subprocess.run(
        [config.FFMPEG_BIN, "-nostdin", "-y", "-i", str(cands[0]), "-vn",
         "-ar", str(config.TARGET_AUDIO_SR), "-ac", str(config.TARGET_AUDIO_CH),
         "-c:a", "aac", "-b:a", f"{config.AUDIO_BITRATE_K}k", str(audio)],
        capture_output=True,
    )
    cands[0].unlink(missing_ok=True)
    if not audio.exists():
        raise PipelineError("Audio transcode failed.")
    return audio


def _whisper_window(audio: Path, win_start: float, win_end: float) -> list[Sentence]:
    win_start = max(0.0, win_start)
    win_len = max(1.0, win_end - win_start)
    tmp = Path(tempfile.mkstemp(suffix=".wav", dir=str(audio.parent))[1])
    try:
        subprocess.run(
            [config.FFMPEG_BIN, "-nostdin", "-y", "-ss", f"{win_start:.3f}", "-t", f"{win_len:.3f}",
             "-i", str(audio), "-ar", "16000", "-ac", "1", str(tmp)],
            capture_output=True,
        )
        model = _get_whisper()
        segments, _ = model.transcribe(str(tmp), word_timestamps=True, beam_size=5)
        words, segs = [], []
        for seg in segments:
            segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
            for w in (seg.words or []):
                words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
    finally:
        tmp.unlink(missing_ok=True)
    if not words:
        return []
    sents = build_sentence_index({"words": words, "segments": segs})
    for s in sents:  # shift to absolute video time
        s.start += win_start
        s.end += win_start
    return sents


def _pick_start(sents: list[Sentence], rough: float, pad: float,
                keep_first: bool = False) -> float:
    """Latest real sentence-start at/just before the rough start (begin at a thought).
    keep_first: the window began at t<=0, so sents[0] is a real start, not a cut fragment."""
    if not sents:
        return rough
    pool = sents if keep_first else sents[1:]
    starts = [s.start for s in pool] or [s.start for s in sents]
    before = [x for x in starts if x <= rough + 1.0]
    if not before:
        return rough              # direction-safe: never move the start LATER than rough+1s
    cand = max(before)
    return cand if abs(cand - rough) <= pad else rough


def _pick_end(sents: list[Sentence], rough: float, pad: float, allow_qe: bool) -> float:
    """Earliest period-terminated end at/just after the rough end (complete the thought)."""
    if not sents:
        return rough
    ends = [s.end for s in sents if s.is_valid_end(allow_qe)]
    if not ends:
        return rough
    # Prefer ends >= rough; fall back to ends >= rough - 1.0
    after = [x for x in ends if x >= rough]
    if not after:
        after = [x for x in ends if x >= rough - 1.0]
    if not after:
        return rough              # direction-safe: never truncate EARLIER than rough-1s
    cand = min(after)
    return cand if abs(cand - rough) <= pad else rough


def _resolve_overlaps(clips: list[dict], min_dur: float, tail_pad: float) -> list[dict]:
    """After independent boundary snapping, ensure clips don't overlap. Trim a clip's
    start to the previous clip's (period) end — a clean sentence boundary — and drop
    any leftover that's too short."""
    clips = sorted(clips, key=lambda c: c["start"])
    out: list[dict] = []
    last_end = -1e9
    for c in clips:
        c = dict(c)
        if c["start"] < last_end:
            c["start"] = round(last_end, 3)
            c["warnings"] = tuple(set(c.get("warnings") or ()) | {"trimmed_start"})
        if c["end"] - c["start"] < min_dur:
            continue
        c["cut_end"] = round(c["end"] + tail_pad, 3)
        out.append(c)
        last_end = c["end"]
    return out


def _refine_one(c: dict, audio: Path, pad: float, allow_qe: bool, tail_pad: float) -> dict:
    """Snap ONE clip's boundaries with its own edge-window Whisper pass. Safe to run
    concurrently over a thread pool: it owns its temp wav (unique mkstemp) and returns a NEW
    clip dict; the Whisper model is a threadsafe singleton (CTranslate2 num_workers); and the
    pysbd segmenter reached via build_sentence_index is thread-LOCAL (sentences.py) — so sibling
    tasks share no mutable state. The returned dict is IDENTICAL to what the serial loop
    computed for this clip."""
    s0, e0 = float(c["start"]), float(c["end"])
    try:
        if e0 - s0 <= 2 * pad + 20:   # short clip → one window covers both ends
            w = _whisper_window(audio, s0 - pad, e0 + pad)
            new_start = _pick_start(w, s0, pad, keep_first=(s0 - pad <= 0.0))
            new_end = _pick_end(w, e0, pad, allow_qe)
        else:
            new_start = _pick_start(_whisper_window(audio, s0 - pad, s0 + pad), s0, pad,
                                    keep_first=(s0 - pad <= 0.0))
            new_end = _pick_end(_whisper_window(audio, e0 - pad, e0 + pad), e0, pad, allow_qe)
        if new_end <= new_start:
            new_start, new_end = s0, e0
    except Exception:
        new_start, new_end = s0, e0
    d = dict(c)
    d["start"] = round(new_start, 3)
    d["end"] = round(new_end, 3)
    d["cut_end"] = round(new_end + tail_pad, 3)
    return d


def refine_clip_boundaries(clips: list[dict], url: str, video_id: str, settings: dict,
                           progress: ProgressCb = None) -> list[dict]:
    allow_qe = bool(settings.get("allow_question_exclaim_ends", False))
    tail_pad = float(settings.get("tail_pad_s", config.DEFAULTS["tail_pad_s"]))
    min_dur = float(settings.get("min_clip_duration_s", config.DEFAULTS["min_clip_duration_s"]))
    pad = config.BOUNDARY_PAD_S
    try:
        audio = _ensure_audio(url, video_id)
    except Exception:
        return clips              # precise pass unavailable → coarse (judged) boundaries stand

    # Latency lever: each clip's edge-window Whisper pass is INDEPENDENT, so run them over a
    # thread pool (sharing the threadsafe singleton model). Threads do only the network/CPU
    # work; every ordering-sensitive step stays SERIAL. Results are collected into a by-INDEX
    # slot regardless of completion order, and the ordering-sensitive post-pass
    # (_resolve_overlaps, whose stable sort-by-start tie-break depends on input order) runs
    # AFTER the pool joins over the ORIGINAL index order → output byte-identical to the serial
    # path. REFINE_WORKERS=1 caps the pool at one worker → the exact serial path.
    n = len(clips)
    total = max(1, n)
    results: list[Optional[dict]] = [None] * n
    workers = max(1, min(config.REFINE_WORKERS, n))
    if workers > 1:
        # Pre-warm the shared Whisper singleton in THIS thread so the pool workers don't race
        # its unguarded lazy init (default TRANSCRIBER=supadata leaves it cold until refine, so
        # multiple threads would otherwise each construct a full model). Once built, concurrent
        # transcribe() is safe (num_workers). The pysbd segmenter is thread-local, so workers
        # build their own cheaply — nothing else is shared.
        try:
            _get_whisper()
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_idx = {pool.submit(_refine_one, c, audio, pad, allow_qe, tail_pad): i
                      for i, c in enumerate(clips)}
        for done, fut in enumerate(as_completed(fut_to_idx), start=1):
            results[fut_to_idx[fut]] = fut.result()   # index slot, NOT completion order
            if progress:
                progress(done / total, f"Refining boundary {done}/{n}")
    out = [d for d in results if d is not None]         # rebuilt in ORIGINAL index order
    return _resolve_overlaps(out, min_dur, tail_pad)
