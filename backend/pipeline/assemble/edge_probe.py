"""Retired video edge probe.

Production never uploads source video or rendered clip fragments to Gemini. The public
entry point remains as a no-op so old callers and cached job settings cannot reactivate it.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ..cut import build_cmd
from . import scoring

_EDGE_WARNINGS = ("starts_mid_sentence_audio", "ends_mid_sentence_audio")


class EdgeVerdict(BaseModel):
    starts_clean_audio: bool = True
    ends_clean_audio: bool = True
    first_words: str = ""
    last_words: str = ""
    evidence: str = ""


EDGE_SYSTEM = (
    "You are an audio-boundary auditor for a short video clip. You are shown the BEGINNING (the "
    "first few seconds) and the ENDING (the last few seconds) of one clip. Listen to the audio. "
    "Decide whether the clip STARTS on a clean boundary — the first words begin a sentence or "
    "thought, not mid-word or mid-sentence — and whether it ENDS on a clean boundary — the last "
    "words complete a sentence or thought, not cut off mid-word or mid-sentence. Report the first "
    "spoken words, the last spoken words, and one short line of evidence. Structured output only."
)


def _cut_segment(video_path: str, start_s: float, dur_s: float) -> Optional[bytes]:
    """Cut a small [start_s, start_s+dur_s] mp4 segment via ffmpeg (cut.build_cmd) into a temp
    file and return its bytes. Fail-soft: returns None on any error / empty output."""
    if not video_path or dur_s <= 0.0:
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "edge.mp4"
            proc = subprocess.run(build_cmd(video_path, float(start_s), float(dur_s), out),
                                  capture_output=True)
            if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
                return None
            return out.read_bytes()
    except Exception:  # noqa: BLE001 — fail-soft
        return None


def _probe_clip(video_path: str, start: float, end: float, media_res,
                probe_s: float) -> Optional[EdgeVerdict]:
    """Retired: Gemini video input is disabled for production."""
    del video_path, start, end, media_res, probe_s
    return None


def _apply_verdict(spec: dict, v: EdgeVerdict) -> None:
    """ADVISORY mapping — warnings + a small final_quality dock, never a kill / Rejection.
    Threads the raw booleans onto the shipped record (surfaced in eval / payload)."""
    spec["starts_clean_audio"] = bool(v.starts_clean_audio)
    spec["ends_clean_audio"] = bool(v.ends_clean_audio)
    warns = set(spec.get("warnings") or ())
    if not v.starts_clean_audio:
        warns.add("starts_mid_sentence_audio")
    if not v.ends_clean_audio:
        warns.add("ends_mid_sentence_audio")
    spec["warnings"] = tuple(warns)
    # Small ADVISORY dock via boundary_score. This runs AFTER the ship gate, so re-scoring can
    # never drop an already-shipped clip; it only makes the concern visible in final_quality.
    if warns & set(_EDGE_WARNINGS) and spec.get("final_quality") is not None:
        spec["boundary_score"] = scoring.boundary_score(spec.get("warnings"))
        if "completeness_score" in spec and "grounding_score" in spec:
            spec["final_quality"] = scoring.quality(
                spec.get("completeness_score", 0.0), spec.get("grounding_score", 0.0),
                spec["boundary_score"], spec.get("priority", 0.0))


def run_edge_probe(clips_spec: list[dict], video_path: str,
                   settings: Optional[dict] = None) -> list[dict]:
    """Return unchanged: production never uploads clip video to Gemini."""
    del video_path, settings
    return clips_spec
