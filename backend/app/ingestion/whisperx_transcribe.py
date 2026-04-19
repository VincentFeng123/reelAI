"""
Phase 4 — selective WhisperX forced alignment.

WhisperX (`whisperx`, wav2vec2-based) produces ±30 ms word timings — tighter
than faster-whisper's ±80–120 ms when audio is noisy or when the caption
track drifts. We do NOT use WhisperX as the default transcription path. The
two supported invocation sites are:

  (a) Low-confidence ingest fallback. When Phase 1's
      `align_words_via_faster_whisper` returns None (match rate < 0.6),
      try `whisperx_align()` as a second-tier refinement before giving up
      and keeping the proportional-word timings.

  (b) Top-k clip refinement. After the global LLM reranker picks its
      shortlist, `clip_whisper_refine._call_whisper` prefers
      `whisperx_words_for_audio()` over the Groq hosted Whisper call for
      the ≤50 ms boundary snap.

Kill switches:
  - `WHISPERX_ENABLED=false`          disables BOTH invocation sites
  - `WHISPERX_FALLBACK_ENABLED=false` disables only (a) — the top-k path
                                       is separately gated by
                                       `WHISPER_CLIP_REFINE_ENABLED`.

The `whisperx` package is imported lazily so environments without it (e.g.
Vercel-slim, test runners) keep working; both helpers return None in that
case and the caller uses its existing fallback.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import IngestTranscriptCue, IngestTranscriptWord

logger = logging.getLogger(__name__)


# Alignment quality regresses across whisperx minor releases (see plan's
# revision history). When upgrading, run the golden-set A/B first.
_PINNED_WHISPERX_VERSION = "3.1.1"

_WHISPERX_ENABLED = os.getenv("WHISPERX_ENABLED", "true").lower() == "true"
_WHISPERX_FALLBACK_ENABLED = os.getenv("WHISPERX_FALLBACK_ENABLED", "true").lower() == "true"
# Railway CPU defaults — keep memory bounded.
_WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
_WHISPERX_BATCH_SIZE = max(1, int(os.getenv("WHISPERX_BATCH_SIZE", "4") or 4))
_WHISPERX_DEVICE = os.getenv("WHISPERX_DEVICE", "cpu")
_WHISPERX_ASR_MODEL = os.getenv("WHISPERX_ASR_MODEL", "small")

_whisperx_module: Any | None = None
_align_model_cache: dict[str, tuple[Any, Any]] = {}
_asr_model_cache: dict[str, Any] = {}


def whisperx_enabled() -> bool:
    """Central kill-switch for BOTH invocation sites."""
    return _WHISPERX_ENABLED


def whisperx_fallback_enabled() -> bool:
    """Kill-switch for site (a) — low-confidence ingest fallback only."""
    return _WHISPERX_ENABLED and _WHISPERX_FALLBACK_ENABLED


def _get_whisperx_module() -> Any | None:
    global _whisperx_module
    if not _WHISPERX_ENABLED:
        return None
    if _whisperx_module is not None:
        return _whisperx_module
    try:
        import whisperx  # type: ignore
    except ImportError:
        logger.info("whisperx package not installed; whisperx alignment disabled")
        return None
    version = getattr(whisperx, "__version__", None)
    if version is not None and str(version) != _PINNED_WHISPERX_VERSION:
        logger.warning(
            "whisperx version %s != pinned %s; alignment may regress — "
            "set WHISPERX_ENABLED=false to disable.",
            version, _PINNED_WHISPERX_VERSION,
        )
    _whisperx_module = whisperx
    return whisperx


def _load_align_model(language: str) -> tuple[Any, Any] | None:
    whisperx = _get_whisperx_module()
    if whisperx is None:
        return None
    lang = (language or "en").split("-")[0].lower()
    cached = _align_model_cache.get(lang)
    if cached is not None:
        return cached
    try:
        model, metadata = whisperx.load_align_model(
            language_code=lang, device=_WHISPERX_DEVICE,
        )
    except Exception as exc:
        logger.info("whisperx.load_align_model(%s) failed: %s", lang, exc)
        return None
    _align_model_cache[lang] = (model, metadata)
    return (model, metadata)


def _load_asr_model() -> Any | None:
    whisperx = _get_whisperx_module()
    if whisperx is None:
        return None
    cached = _asr_model_cache.get(_WHISPERX_ASR_MODEL)
    if cached is not None:
        return cached
    try:
        model = whisperx.load_model(
            _WHISPERX_ASR_MODEL,
            _WHISPERX_DEVICE,
            compute_type=_WHISPERX_COMPUTE_TYPE,
        )
    except Exception as exc:
        logger.info("whisperx.load_model(%s) failed: %s", _WHISPERX_ASR_MODEL, exc)
        return None
    _asr_model_cache[_WHISPERX_ASR_MODEL] = model
    return model


def whisperx_align(
    audio_path: Path,
    caption_cues: list[IngestTranscriptCue],
    *,
    language: str,
) -> list[IngestTranscriptCue] | None:
    """
    Forced alignment of existing caption text against wav2vec2.

    Mirrors `align_words_via_faster_whisper` so callers can substitute it
    when Phase-1 alignment came up short. Caption text is preserved; only
    word-level start/end timings change, and `word_source` becomes
    `"whisperx"` so the sentence splitter can credit it appropriately.

    Returns None on any failure — disabled, unavailable package, missing
    language model, alignment exception, or empty result.
    """
    if not _WHISPERX_ENABLED or not caption_cues:
        return None
    whisperx = _get_whisperx_module()
    if whisperx is None:
        return None
    aligner = _load_align_model(language)
    if aligner is None:
        return None
    align_model, align_meta = aligner

    segments_in: list[dict[str, Any]] = []
    keep_idx: list[int] = []
    for i, c in enumerate(caption_cues):
        text = (c.text or "").strip()
        if not text:
            continue
        segments_in.append({"start": float(c.start), "end": float(c.end), "text": text})
        keep_idx.append(i)
    if not segments_in:
        return None

    try:
        audio = whisperx.load_audio(str(audio_path))
    except Exception as exc:
        logger.info("whisperx.load_audio failed: %s", exc)
        return None
    try:
        result = whisperx.align(
            segments_in,
            align_model,
            align_meta,
            audio,
            _WHISPERX_DEVICE,
            return_char_alignments=False,
        )
    except Exception as exc:
        logger.info("whisperx.align raised: %s", exc)
        return None

    aligned_segments = result.get("segments") if isinstance(result, dict) else None
    if not aligned_segments or len(aligned_segments) != len(segments_in):
        logger.info(
            "whisperx returned %d segments (expected %d); dropping alignment",
            len(aligned_segments) if aligned_segments else 0, len(segments_in),
        )
        return None

    aligned_by_cue_idx: dict[int, IngestTranscriptCue] = {}
    for seg_idx, seg in enumerate(aligned_segments):
        cue_idx = keep_idx[seg_idx]
        cue = caption_cues[cue_idx]
        seg_words = seg.get("words") if isinstance(seg, dict) else None
        if not seg_words:
            aligned_by_cue_idx[cue_idx] = cue  # keep original when empty
            continue
        new_words: list[IngestTranscriptWord] = []
        for w in seg_words:
            try:
                ws = float(w.get("start", cue.start))
                we = float(w.get("end", cue.end))
            except (TypeError, ValueError):
                continue
            text = str(w.get("word", "")).strip()
            if not text or we <= ws:
                continue
            score = w.get("score")
            conf: float | None
            try:
                conf = max(0.0, min(1.0, float(score))) if score is not None else None
            except (TypeError, ValueError):
                conf = None
            new_words.append(IngestTranscriptWord(
                start=ws, end=max(we, ws + 0.01), text=text, confidence=conf,
            ))
        if not new_words:
            aligned_by_cue_idx[cue_idx] = cue
            continue
        aligned_by_cue_idx[cue_idx] = IngestTranscriptCue(
            start=cue.start, end=cue.end, text=cue.text,
            words=new_words, word_source="whisperx",
        )

    aligned_cues = [
        aligned_by_cue_idx.get(i, caption_cues[i]) for i in range(len(caption_cues))
    ]
    upgraded = sum(1 for c in aligned_cues if c.word_source == "whisperx")
    logger.info(
        "whisperx alignment applied: cues=%d upgraded=%d language=%s",
        len(aligned_cues), upgraded, language,
    )
    if upgraded == 0:
        return None
    return aligned_cues


@dataclass
class WhisperXWord:
    """Raw word timing from WhisperX — used by the top-k clip refiner.

    Kept intentionally similar to `clip_whisper_refine.WhisperWord` so
    callers can map between them without an adapter."""
    text: str
    start: float
    end: float


def whisperx_words_for_audio(
    audio_path: Path,
    *,
    language: str = "en",
) -> list[WhisperXWord]:
    """
    Transcribe + align a short audio clip and return raw word timings.

    Used by `clip_whisper_refine._call_whisper` as the WhisperX alternative
    to Groq's hosted Whisper endpoint. Timings are relative to the clip's
    t=0 — the caller adds the download offset to get absolute video time.
    """
    if not _WHISPERX_ENABLED:
        return []
    whisperx = _get_whisperx_module()
    if whisperx is None:
        return []
    asr = _load_asr_model()
    if asr is None:
        return []
    aligner = _load_align_model(language)
    if aligner is None:
        return []
    align_model, align_meta = aligner
    try:
        audio = whisperx.load_audio(str(audio_path))
    except Exception as exc:
        logger.info("whisperx.load_audio failed for clip: %s", exc)
        return []
    try:
        asr_result = asr.transcribe(audio, batch_size=_WHISPERX_BATCH_SIZE)
    except Exception as exc:
        logger.info("whisperx asr.transcribe raised for clip: %s", exc)
        return []
    segments = asr_result.get("segments") if isinstance(asr_result, dict) else None
    if not segments:
        return []
    try:
        aligned = whisperx.align(
            segments, align_model, align_meta, audio,
            _WHISPERX_DEVICE, return_char_alignments=False,
        )
    except Exception as exc:
        logger.info("whisperx.align raised for clip: %s", exc)
        return []
    aligned_segments = aligned.get("segments") if isinstance(aligned, dict) else None
    if not aligned_segments:
        return []

    out: list[WhisperXWord] = []
    for seg in aligned_segments:
        words = seg.get("words") if isinstance(seg, dict) else None
        if not words:
            continue
        for w in words:
            try:
                start = float(w.get("start"))
                end = float(w.get("end"))
            except (TypeError, ValueError):
                continue
            text = str(w.get("word", "")).strip()
            if not text or end <= start:
                continue
            out.append(WhisperXWord(text=text, start=start, end=end))
    return out


__all__ = [
    "WhisperXWord",
    "whisperx_enabled",
    "whisperx_fallback_enabled",
    "whisperx_align",
    "whisperx_words_for_audio",
]
