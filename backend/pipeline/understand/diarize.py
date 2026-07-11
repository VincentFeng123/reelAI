"""Speaker diarization via pyannote (optional; degrades to single/None speaker).

Attaches speaker turns to the audio so units can carry a ``speaker`` label. This sharpens the
``continues`` edge (same-speaker runs), enables the interview family (attribute a question vs its
answer to different people), and is entirely optional: if ``HF_TOKEN`` is unset, ``pyannote.audio``
isn't installed, or the gated model terms aren't accepted, ``available()`` is False and every entry
point returns empty so the pipeline runs exactly as before (single speaker).

Enabling it (one-time): ``pip install "pyannote.audio>=3.1"``, set ``HF_TOKEN`` in ``.env`` (read
scope), and accept the gated terms at hf.co/pyannote/speaker-diarization-3.1 and
hf.co/pyannote/segmentation-3.0. Then run with ``diarization`` on (per-job setting or ``DIARIZATION=1``).
"""
from __future__ import annotations

from typing import Callable, Optional

from ... import config
from .models import SpeakerTurn

ProgressCb = Optional[Callable[[float, str], None]]

_pipeline = None                                           # cached pyannote Pipeline (heavy to build)


def available() -> bool:
    """True only if a token is set AND pyannote.audio imports (dep + gated terms are user setup)."""
    if not config.HF_TOKEN:
        return False
    try:
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(config.DIARIZATION_MODEL, use_auth_token=config.HF_TOKEN)
    return _pipeline


def diarize(audio_path: str, video_id: str, progress: ProgressCb = None) -> list[SpeakerTurn]:
    """Speaker turns for ``audio_path`` (16 kHz mono from download()); [] if unavailable/failed."""
    if not available() or not audio_path:
        return []
    try:
        annotation = _get_pipeline()(audio_path)           # blocking; run in the executor thread
        turns = [SpeakerTurn(start=float(seg.start), end=float(seg.end), speaker=str(spk))
                 for seg, _, spk in annotation.itertracks(yield_label=True)]
    except Exception:
        return []                                          # honor the "[] on failure" contract fully
    if progress:
        progress(1.0, f"Diarized {len(turns)} turns")
    return turns


def assign_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> Optional[str]:
    """The dominant speaker over [start,end] by overlap; nearest-by-midpoint if no overlap."""
    if not turns:
        return None
    best, best_ov = None, 0.0
    for tr in turns:
        ov = max(0.0, min(end, tr.end) - max(start, tr.start))
        if ov > best_ov:
            best, best_ov = tr.speaker, ov
    if best is None:                                       # no overlap → closest turn by midpoint
        mid = (start + end) / 2.0
        best = min(turns, key=lambda tr: abs(((tr.start + tr.end) / 2.0) - mid)).speaker
    return best
