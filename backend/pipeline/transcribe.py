"""Groq Whisper transcription → punctuated, word-level timestamps.

Audio ≤ 25 MB (Groq free-tier cap) → one request. Larger audio → split into
time-chunks (with overlap), transcribe sequentially, and merge with corrected
time offsets, de-duplicating the overlap region. Cached to transcript.json.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .. import config
from ..errors import PipelineError
from ..groq_client import transcribe_audio

ProgressCb = Optional[Callable[[float, str], None]]

# ── faster-whisper (local) ───────────────────────────────────────────────────
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # num_workers lets CTranslate2 run concurrent transcribe() calls in TRUE parallel
        # (boundary REFINE fans out one edge-window pass per clip over REFINE_WORKERS threads
        # on this shared singleton). It is a pure throughput knob — it changes NOTHING about any
        # single transcription's output — and REFINE_WORKERS=1 → num_workers=1 = the prior model.
        # CPU-bound int8 model, so this is capped modestly (REFINE_WORKERS default 4).
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE,
            num_workers=max(1, config.REFINE_WORKERS),
        )
    return _whisper_model


def _transcribe_local(audio_path: str, lang: str, emit) -> dict:
    model = _get_whisper()
    emit(0.05, f"Transcribing locally (Whisper {config.WHISPER_MODEL})…")
    segments, info = model.transcribe(
        audio_path, language=lang or None, word_timestamps=True, beam_size=5
    )
    total = float(getattr(info, "duration", 0.0) or 0.0)
    words: list[dict] = []
    segs: list[dict] = []
    for seg in segments:  # generator → runs inference lazily
        segs.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
        for w in (seg.words or []):
            words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
        if total:
            emit(min(0.99, float(seg.end) / total), "Transcribing locally…")
    return {
        "text": " ".join(s["text"].strip() for s in segs),
        "duration": total,
        "words": words,
        "segments": segs,
    }


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _extract_chunk(audio_path: str, start: float, length: float, out: Path) -> None:
    cmd = [
        config.FFMPEG_BIN, "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", audio_path,
        "-ar", str(config.TARGET_AUDIO_SR), "-ac", str(config.TARGET_AUDIO_CH),
        "-c:a", "aac", "-b:a", f"{config.AUDIO_BITRATE_K}k", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        raise PipelineError(f"audio chunking failed: {tail[-1] if tail else proc.returncode}")


def _transcribe_chunked(audio_path: str, size_mb: float, lang: str, emit) -> dict:
    dur = _probe_duration(audio_path)
    if dur <= 0:
        raise PipelineError("Could not determine audio duration for chunking.")
    num = max(2, math.ceil(size_mb / (config.GROQ_MAX_FILE_MB * 0.85)))
    seg = dur / num
    overlap = config.AUDIO_CHUNK_OVERLAP_S
    tmp = Path(audio_path).parent

    words: list[dict] = []
    segments: list[dict] = []
    last_w_end = -1.0
    last_s_end = -1.0

    for k in range(num):
        start = k * seg
        length = seg + (overlap if k < num - 1 else 0.0)
        chunk_path = tmp / f"_chunk_{k}.m4a"
        _extract_chunk(audio_path, start, length, chunk_path)
        try:
            r = transcribe_audio(str(chunk_path), language=lang)
        finally:
            chunk_path.unlink(missing_ok=True)

        for w in r.get("words") or []:
            aw = {
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)) + start,
                "end": float(w.get("end", 0.0)) + start,
            }
            if aw["start"] >= last_w_end - 1e-3:   # drop overlap-region duplicates
                words.append(aw)
                last_w_end = max(last_w_end, aw["end"])
        for s in r.get("segments") or []:
            as_ = {
                "start": float(s.get("start", 0.0)) + start,
                "end": float(s.get("end", 0.0)) + start,
                "text": s.get("text", ""),
            }
            if as_["start"] >= last_s_end - 1e-3:
                segments.append(as_)
                last_s_end = max(last_s_end, as_["end"])

        emit((k + 1) / num, f"Transcribing chunk {k + 1}/{num}…")

    return {
        "text": " ".join(s["text"].strip() for s in segments),
        "duration": dur,
        "words": words,
        "segments": segments,
    }


def transcribe_supadata(url: str, video_id: str, settings: dict, progress: ProgressCb = None) -> dict:
    """Fetch a timestamped transcript from Supadata (no audio download).

    Synthesizes per-word times (proportional within each chunk) so the existing
    sentence index/alignment works unchanged; also keeps the raw chunks for the
    no-punctuation fallback.
    """
    def emit(frac: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    cache = config.WORK_DIR / video_id / "transcript.json"
    if cache.exists():
        emit(1.0, "Using cached transcript")
        return json.loads(cache.read_text(encoding="utf-8"))

    from ..supadata_client import fetch_transcript

    emit(0.2, "Fetching transcript (Supadata)…")
    chunks = fetch_transcript(
        url, settings.get("language", "en"), chunk_size=config.SUPADATA_CHUNK_SIZE
    )

    words: list[dict] = []
    segments: list[dict] = []
    for ch in chunks:
        segments.append({"start": ch["start"], "end": ch["end"], "text": ch["text"]})
        toks = ch["text"].split()
        n = max(1, len(toks))
        dur = max(0.01, ch["end"] - ch["start"])
        for j, tk in enumerate(toks):
            words.append({
                "word": tk,
                "start": ch["start"] + dur * j / n,
                "end": ch["start"] + dur * (j + 1) / n,
            })

    result = {
        "text": " ".join(s["text"] for s in segments),
        "duration": segments[-1]["end"] if segments else 0.0,
        "words": words,
        "segments": segments,
        "source": "supadata",
        "chunks": chunks,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    emit(1.0, "Transcript ready")
    return result


def transcribe(audio_path: str, video_id: str, settings: dict, progress: ProgressCb = None) -> dict:
    """Return {'text','duration','words':[{word,start,end}],'segments':[{start,end,text}]}."""
    def emit(frac: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    cache = config.WORK_DIR / video_id / "transcript.json"
    if cache.exists():
        emit(1.0, "Using cached transcript")
        return json.loads(cache.read_text(encoding="utf-8"))

    lang = settings.get("language", "en")

    if config.TRANSCRIBER == "faster_whisper":
        result = _transcribe_local(audio_path, lang, emit)
    else:
        size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        if size_mb <= config.GROQ_MAX_FILE_MB:
            emit(0.1, "Transcribing with Groq Whisper…")
            result = transcribe_audio(audio_path, language=lang)
        else:
            emit(0.05, f"Long audio ({size_mb:.0f} MB) — chunking for Groq…")
            result = _transcribe_chunked(audio_path, size_mb, lang, emit)

    if not result.get("words"):
        raise PipelineError(
            "Transcription returned no word-level timestamps (cannot place cuts precisely)."
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    emit(1.0, "Transcription complete")
    return result
