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
import wave
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config
from ..errors import PipelineError
from . import download as download_mod
from .sentences import Sentence, build_sentence_index
from .transcribe import _get_refine_whisper

Pick = namedtuple("Pick", ["time", "flags", "satisfied"])

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


def _whisper_window(audio: Path, win_start: float, win_end: float) -> "tuple[list[Sentence], Path | None]":
    win_start = max(0.0, win_start)
    win_len = max(1.0, win_end - win_start)
    tmp = Path(tempfile.mkstemp(suffix=".wav", dir=str(audio.parent))[1])
    ok = False
    try:
        subprocess.run(
            [config.FFMPEG_BIN, "-nostdin", "-y", "-ss", f"{win_start:.3f}", "-t", f"{win_len:.3f}",
             "-i", str(audio), "-ar", "16000", "-ac", "1", str(tmp)],
            capture_output=True,
        )
        model = _get_refine_whisper()
        kw = dict(word_timestamps=True, beam_size=5, temperature=0.0,
                  condition_on_previous_text=False)
        if config.REFINE_VAD:
            kw["vad_filter"] = True
            kw["vad_parameters"] = dict(speech_pad_ms=200)
        segments, _ = model.transcribe(str(tmp), **kw)
        words, segs = [], []
        for seg in segments:
            segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
            for w in (seg.words or []):
                words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
        ok = True
    finally:
        if not ok:
            tmp.unlink(missing_ok=True)
    if not words:
        tmp.unlink(missing_ok=True)
        return [], None
    sents = build_sentence_index({"words": words, "segments": segs})
    for s in sents:  # shift to absolute video time
        s.start += win_start
        s.end += win_start
    return sents, tmp        # caller owns tmp cleanup (energy step reads it first)


def _energy_min_snap(wav_path, win_start: float, a: float, b: float,
                     frame_ms: int = 10) -> "float | None":
    """Absolute time of the lowest-RMS ``frame_ms`` frame within ``[a, b]`` — the quietest instant
    in the pause. ``win_start`` is the wav's absolute start time. Returns None on a bad/short read
    so the caller keeps its pad/midpoint fallback (never raises)."""
    if wav_path is None or b <= a:
        return None
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    except Exception:
        return None
    if samples.size == 0:
        return None
    frame = max(1, int(sr * frame_ms / 1000))
    lo = max(0, int((a - win_start) * sr))
    hi = min(samples.size, int((b - win_start) * sr))
    if hi - lo < frame:
        return None
    best_i, best_rms = lo, None
    for i in range(lo, hi - frame + 1, frame):
        seg = samples[i:i + frame]
        rms = float(np.sqrt(np.mean(seg * seg)))
        if best_rms is None or rms < best_rms:
            best_rms, best_i = rms, i
    return win_start + (best_i + frame / 2) / sr


def _gap_before(sents, idx: int) -> "tuple[float, float] | None":
    """(prev.end, S.start) for the sentence at idx, or None if idx is the window's first sentence."""
    if idx <= 0 or idx >= len(sents):
        return None
    return sents[idx - 1].end, sents[idx].start


def _snap_start_cut(s_start: float, prev_end: float, lead_pad: float, energy_fn) -> "tuple[float, tuple]":
    """Cut into the gap BEFORE s_start: max(s_start-lead_pad, midpoint(prev_end, s_start)), then
    energy-snap within [that, s_start]. Never < prev_end, never > s_start."""
    mid = (prev_end + s_start) / 2.0
    pad_cut = max(s_start - lead_pad, mid)
    cut = pad_cut
    if energy_fn is not None:
        snapped = energy_fn(pad_cut, s_start)
        if snapped is not None:
            cut = min(max(snapped, pad_cut), s_start)
    return max(prev_end, min(cut, s_start)), ()


def _pick_start(sents: list[Sentence], rough: float, pad: float,
                keep_first: bool = False, *, lead_pad: float = 0.06,
                gap_min: float = 0.12, energy_fn=None) -> Pick:
    """Begin at a thought's onset, cutting into the leading inter-word gap (never into the prev
    word). Direction-safe: never chooses a start later than rough+1s. The START never re-selects
    an earlier sentence — if the previous word isn't visible, it asks _refine_start to grow the
    window (satisfied=False)."""
    if not sents:
        return Pick(rough, (), True)
    # candidate starts: exclude the window's first sentence as a fragment unless keep_first
    cand = [(i, s.start) for i, s in enumerate(sents) if (keep_first or i >= 1)]
    before = [(i, x) for (i, x) in cand if x <= rough + 1.0]
    if not before:
        return Pick(rough, (), True)                 # direction-safe: keep rough
    idx, s_start = max(before, key=lambda t: t[1])
    if abs(s_start - rough) > pad:
        return Pick(rough, (), True)                 # nearest onset too far → keep rough
    gap = _gap_before(sents, idx)
    # Determine if the previous word is visible/reliable.
    # When S=sents[1] (idx==1) and not keep_first, sents[0] is the window's first entry.
    # If sents[0] is an unterminated fragment (terminator=""), its .end is mid-speech and
    # cannot serve as a reliable inter-word boundary → treat prev as unseen.
    # NOTE: use .terminator (not .ends_with_period) because "" in ".?!" is True in Python.
    prev_unseen = gap is None
    if (not prev_unseen and idx == 1 and not keep_first
            and not sents[0].terminator):
        prev_unseen = True
    if prev_unseen:
        cut = max(0.0, s_start - lead_pad)
        if energy_fn is not None:
            snapped = energy_fn(cut, s_start)
            if snapped is not None:
                cut = min(max(snapped, cut), s_start)
        if keep_first:
            return Pick(round(cut, 3), (), True)     # window at t=0 → real onset, no prev needed
        return Pick(round(cut, 3), ("start_prev_unseen",), False)   # prev not visible → grow back
    prev_end, _ = gap
    cut, flags = _snap_start_cut(s_start, prev_end, lead_pad, energy_fn)
    return Pick(round(cut, 3), flags, True)


def _gap_after(sents, idx: int) -> "tuple[float, float] | None":
    """(E.end, next.start) for the sentence at idx, or None if idx is the window's last sentence."""
    if idx < 0 or idx + 1 >= len(sents):
        return None
    return sents[idx].end, sents[idx + 1].start


def _snap_end_cut(e_end: float, next_start: float, tail_pad: float, energy_fn) -> "tuple[float, tuple]":
    """Cut into the gap AFTER e_end: min(e_end+tail_pad, midpoint(e_end, next_start)), then
    energy-snap within [e_end, that]. Never > next_start, never < e_end."""
    mid = (e_end + next_start) / 2.0
    pad_cut = min(e_end + tail_pad, mid)
    cut = pad_cut
    if energy_fn is not None:
        snapped = energy_fn(e_end, pad_cut)
        if snapped is not None:
            cut = min(max(snapped, e_end), pad_cut)
    return min(next_start, max(cut, e_end)), ()


def _pick_end(sents: list[Sentence], rough: float, pad: float, allow_qe: bool, *,
              tail_pad: float = 0.15, gap_min: float = 0.12,
              end_extend_max: float = 8.0, energy_fn=None) -> Pick:
    """Complete the thought, cutting into the trailing inter-word gap (never into the next word).
    HYBRID (handoff §8): tight cut at the chosen complete sentence when it has a usable trailing
    gap; only nudge forward to a later gap within end_extend_max; else keep tight + flag. Direction-
    safe: never truncates earlier than rough-1s. satisfied=False only when no valid end with a
    MEASURABLE gap is in the window (→ _refine_end grows)."""
    if not sents:
        return Pick(rough, (), False)
    valids = [i for i, s in enumerate(sents) if s.is_valid_end(allow_qe)]
    at_after = [i for i in valids if sents[i].end >= rough] or \
               [i for i in valids if sents[i].end >= rough - 1.0]
    if not at_after:
        return Pick(rough, (), False)                    # no valid end at all → grow
    at_after.sort(key=lambda i: sents[i].end)
    e_idx = at_after[0]
    gap = _gap_after(sents, e_idx)
    if gap is None:
        return Pick(round(sents[e_idx].end + tail_pad, 3), (), False)   # last in window → grow
    e_end, nxt = gap
    if (nxt - e_end) >= gap_min:
        cut, flags = _snap_end_cut(e_end, nxt, tail_pad, energy_fn)
        return Pick(round(cut, 3), flags, True)          # tight cut in a real gap
    # hybrid: look forward within budget for a later end WITH a usable gap
    for i in at_after:
        if sents[i].end > sents[e_idx].end + end_extend_max:
            break
        g = _gap_after(sents, i)
        if g and (g[1] - g[0]) >= gap_min:
            cut, _ = _snap_end_cut(g[0], g[1], tail_pad, energy_fn)
            return Pick(round(cut, 3), ("end_extended",), True)
    # none in budget → best-available tight cut at E, flagged
    cut, _ = _snap_end_cut(e_end, nxt, tail_pad, energy_fn)
    return Pick(round(cut, 3), ("tight_end_no_gap",), True)


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


def _refine_end(audio, e0, pad, allow_qe, *, tail_pad, gap_min, end_extend_max,
                max_search, max_clip_end) -> Pick:
    grow = pad
    last = Pick(e0, (), False)
    while True:
        win_start = max(0.0, e0 - pad)
        win_end = min(e0 + grow, e0 + max_search, max_clip_end)
        sents, wav = _whisper_window(audio, win_start, win_end)
        try:
            energy_fn = (lambda a, b: _energy_min_snap(wav, win_start, a, b)) if wav else None
            last = _pick_end(sents, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                             end_extend_max=end_extend_max, energy_fn=energy_fn)
        finally:
            if wav is not None:
                wav.unlink(missing_ok=True)
        if last.satisfied:
            return last
        if win_end >= e0 + max_search - 1e-6 or win_end >= max_clip_end - 1e-6:
            break
        grow *= 2
    return Pick(last.time, tuple(sorted(set(last.flags) | {"boundary_search_exhausted"})), False)


def _refine_start(audio, s0, pad, *, lead_pad, gap_min, max_search) -> Pick:
    grow = pad
    last = Pick(s0, (), False)
    while True:
        win_start = max(0.0, s0 - grow)
        win_end = s0 + pad
        sents, wav = _whisper_window(audio, win_start, win_end)
        try:
            energy_fn = (lambda a, b: _energy_min_snap(wav, win_start, a, b)) if wav else None
            last = _pick_start(sents, s0, pad, keep_first=(win_start <= 0.0),
                               lead_pad=lead_pad, gap_min=gap_min, energy_fn=energy_fn)
        finally:
            if wav is not None:
                wav.unlink(missing_ok=True)
        if last.satisfied:
            return last
        if win_start <= 0.0 or (s0 - win_start) >= max_search - 1e-6:
            break
        grow *= 2
    return Pick(last.time, tuple(sorted(set(last.flags) | {"start_prev_unseen"})), False)


def _refine_one(c, audio, pad, allow_qe, tail_pad, lead_pad, gap_min, end_extend_max,
                max_search, max_clip_dur) -> dict:
    """Snap ONE clip's boundaries using window-extending Whisper passes. Safe to run
    concurrently over a thread pool: each call owns its temp wavs (unique mkstemp), returns a
    NEW clip dict, and shares no mutable state with sibling tasks."""
    s0, e0 = float(c["start"]), float(c["end"])
    flags: tuple = ()
    try:
        max_clip_end = s0 + max_clip_dur
        if e0 - s0 <= 2 * pad + 20:                       # short clip → try one combined window
            win_start = max(0.0, s0 - pad)
            sents, wav = _whisper_window(audio, s0 - pad, e0 + pad)
            try:
                energy_fn = (lambda a, b: _energy_min_snap(wav, win_start, a, b)) if wav else None
                sp = _pick_start(sents, s0, pad, keep_first=(win_start <= 0.0),
                                 lead_pad=lead_pad, gap_min=gap_min, energy_fn=energy_fn)
                ep = _pick_end(sents, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                               end_extend_max=end_extend_max, energy_fn=energy_fn)
            finally:
                if wav is not None:
                    wav.unlink(missing_ok=True)
            if not sp.satisfied:
                sp = _refine_start(audio, s0, pad, lead_pad=lead_pad, gap_min=gap_min,
                                   max_search=max_search)
            if not ep.satisfied:
                ep = _refine_end(audio, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                                 end_extend_max=end_extend_max, max_search=max_search,
                                 max_clip_end=max_clip_end)
        else:
            sp = _refine_start(audio, s0, pad, lead_pad=lead_pad, gap_min=gap_min,
                               max_search=max_search)
            ep = _refine_end(audio, e0, pad, allow_qe, tail_pad=tail_pad, gap_min=gap_min,
                             end_extend_max=end_extend_max, max_search=max_search,
                             max_clip_end=max_clip_end)
        new_start, new_end = sp.time, ep.time
        flags = tuple(sorted(set(sp.flags) | set(ep.flags)))
        if new_end <= new_start:
            new_start, new_end, flags = s0, e0, tuple(sorted(set(flags) | {"refine_degenerate"}))
    except Exception:
        new_start, new_end, flags = s0, e0, ()
    d = dict(c)
    d["start"] = round(new_start, 3)
    d["end"] = round(new_end, 3)
    d["cut_end"] = round(new_end + tail_pad, 3)
    if flags:
        d["warnings"] = tuple(sorted(set(d.get("warnings") or ()) | set(flags)))
    return d


def refine_clip_boundaries(clips: list[dict], url: str, video_id: str, settings: dict,
                           progress: ProgressCb = None) -> list[dict]:
    allow_qe = bool(settings.get("allow_question_exclaim_ends", False))
    tail_pad = float(settings.get("tail_pad_s", config.DEFAULTS["tail_pad_s"]))
    lead_pad = float(settings.get("lead_pad_s", config.DEFAULTS["lead_pad_s"]))
    min_dur = float(settings.get("min_clip_duration_s", config.DEFAULTS["min_clip_duration_s"]))
    max_dur = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    gap_min = config.SILENCE_MIN_GAP_S
    end_extend_max = config.END_EXTEND_MAX_S
    max_search = config.MAX_BOUNDARY_SEARCH_S
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
        # Pre-warm the shared refine Whisper singleton in THIS thread so the pool workers don't
        # race its unguarded lazy init (default TRANSCRIBER=supadata leaves it cold until refine,
        # so multiple threads would otherwise each construct a full model). Once built, concurrent
        # transcribe() is safe (num_workers). The pysbd segmenter is thread-local, so workers
        # build their own cheaply — nothing else is shared.
        try:
            _get_refine_whisper()
        except Exception:
            pass
    args = (pad, allow_qe, tail_pad, lead_pad, gap_min, end_extend_max, max_search, max_dur)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_idx = {pool.submit(_refine_one, c, audio, *args): i for i, c in enumerate(clips)}
        for done, fut in enumerate(as_completed(fut_to_idx), start=1):
            results[fut_to_idx[fut]] = fut.result()   # index slot, NOT completion order
            if progress:
                progress(done / total, f"Refining boundary {done}/{n}")
    out = [d for d in results if d is not None]         # rebuilt in ORIGINAL index order
    return _resolve_overlaps(out, min_dur, tail_pad)
