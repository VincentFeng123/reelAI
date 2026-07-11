"""VID2 — edge probe (Tier 1 video judge): advisory audio-boundary check on shipped clips.

For each shipped survivor we cut its FIRST ~N seconds and LAST ~N seconds LOCALLY from the
source video (reusing ``cut.build_cmd``) and ask a video-capable Gemini — inline mp4 bytes at
LOW media resolution — whether the clip STARTS and ENDS on clean audio (not mid-sentence /
mid-word). This is the cheapest video-judge tier (~$0.0001-0.0004/clip, no Files upload) and
catches the F7 mid-sentence-audio blind spot the transcript-anchored boundary snap can miss.

ADVISORY ONLY (mirrors validate.py's W25-E single_idea): a False verdict adds a WARNING
('starts_mid_sentence_audio' / 'ends_mid_sentence_audio') and a small final_quality dock — it
NEVER kills a clip and NEVER creates a Rejection, so it cannot touch validate.py's text kill
gate (``unverified_kill`` stays 0 trivially). Gemini-only (Groq has no video input), so it
bypasses llm.py/llm_json and calls gemini_client directly, exactly as vision.py does.

Fail-soft: any ffmpeg/LLM error on a clip leaves that clip untouched (no warning added) and
never raises into the pipeline — same contract as write_run_artifacts.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ... import config
from ...gemini_client import (generate_json_video, media_resolution_from_name,
                              text_part, video_part_inline)
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
    """Cut the head + tail, send ONE inline video judge call, return the parsed EdgeVerdict.
    Returns None if both cuts fail (nothing to judge). May raise on the LLM call — the caller
    guards it (any error → that clip is left untouched)."""
    head_dur = min(probe_s, max(0.0, end - start))
    head = _cut_segment(video_path, start, head_dur)
    tail_start = max(start, end - probe_s)
    tail = _cut_segment(video_path, tail_start, max(0.0, end - tail_start))
    if head is None and tail is None:
        return None

    parts = [text_part(f"BEGINNING of the clip (first ~{probe_s:.0f}s of audio):")]
    if head is not None:
        parts.append(video_part_inline(head, media_resolution=media_res))
    parts.append(text_part(f"ENDING of the clip (last ~{probe_s:.0f}s of audio):"))
    if tail is not None:
        parts.append(video_part_inline(tail, media_resolution=media_res))

    raw = generate_json_video(EDGE_SYSTEM, parts, EdgeVerdict, media_resolution=media_res)
    return EdgeVerdict.model_validate_json(raw)


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
    """Advisory edge probe over the shipped survivors — mutates each spec's warnings in place
    (never kills, never raises). Returns clips_spec (mutated) for call-site convenience."""
    try:
        if not clips_spec or not video_path:
            return clips_spec
        media_res = media_resolution_from_name(config.VIDEO_MEDIA_RESOLUTION)
        probe_s = float(config.EDGE_PROBE_SECONDS)
        for spec in clips_spec:
            try:
                v = _probe_clip(video_path, float(spec["start"]), float(spec["end"]),
                                media_res, probe_s)
            except Exception:  # noqa: BLE001 — one clip's LLM/parse error must not spread
                v = None
            if v is not None:
                _apply_verdict(spec, v)
    except Exception:  # noqa: BLE001 — never raise into the pipeline
        pass
    return clips_spec
