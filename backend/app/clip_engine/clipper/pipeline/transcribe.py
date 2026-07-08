"""Transcription dispatch: Supadata (default) or faster-whisper (local fallback).

Cached to transcript.json.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from .. import config
from ..errors import PipelineError

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


# Dedicated singleton for the boundary-REFINE pass, which uses a (usually larger) model than full
# transcription for more precise word timestamps. Keyed by REFINE_WHISPER_MODEL. When it equals
# WHISPER_MODEL we reuse the full-transcription singleton so the model loads only once. Threadsafe
# once built (CTranslate2 num_workers); refine_clip_boundaries pre-warms it before its thread pool.
_refine_whisper_model = None


def _get_refine_whisper():
    global _refine_whisper_model
    if config.REFINE_WHISPER_MODEL == config.WHISPER_MODEL:
        return _get_whisper()
    if _refine_whisper_model is None:
        from faster_whisper import WhisperModel
        _refine_whisper_model = WhisperModel(
            config.REFINE_WHISPER_MODEL, device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE, num_workers=max(1, config.REFINE_WORKERS),
        )
    return _refine_whisper_model


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

    if config.TRANSCRIBER in ("faster_whisper", "local"):
        result = _transcribe_local(audio_path, lang, emit)
    else:
        raise PipelineError(
            f"Unknown TRANSCRIBER value '{config.TRANSCRIBER}'. "
            "Set TRANSCRIBER=faster_whisper for local transcription."
        )

    if not result.get("words"):
        raise PipelineError(
            "Transcription returned no word-level timestamps (cannot place cuts precisely)."
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    emit(1.0, "Transcription complete")
    return result
