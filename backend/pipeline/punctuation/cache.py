"""On-disk caching for punctuation.

Two levels, both under ``work/<video_id>/``:
- ``punctuation.json`` — the whole ``PunctuationResult``, gated on schema_version + a
  ``transcript_fingerprint`` + ``token_count``. The strict gate matters because token ids are
  positional: a re-transcode under the same ``video_id`` must MISS, not silently mis-map ids.
- ``punctuation/chunks/<hash>.json`` — one accepted chunk annotation, so a partial failure/retry
  never recomputes a good chunk.

All disk access is best-effort; any failure degrades to "no cache" and the stage recomputes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from ... import config
from .types import (
    Annotation,
    PunctuationArtifact,
    PunctuationResult,
    TimedWord,
    TranscriptChunk,
)


def transcript_fingerprint(words: list[TimedWord], source: str) -> str:
    n = len(words)
    dur = round(float(words[-1].end), 1) if words else 0.0
    head = "|".join(f"{w.word.strip()}:{round(w.start, 2)}" for w in words[:8])
    tail = "|".join(f"{w.word.strip()}:{round(w.end, 2)}" for w in words[-8:])
    return hashlib.sha1(f"{source}|{n}|{dur}|{head}|{tail}".encode("utf-8")).hexdigest()


def _video_dir(video_id: str) -> Path:
    return config.WORK_DIR / video_id


def _artifact_path(video_id: str) -> Path:
    return _video_dir(video_id) / "punctuation.json"


def load_artifact(video_id: str, words: list[TimedWord], source: str, model: str,
                  prompt_version: str) -> Optional[PunctuationResult]:
    if not video_id:
        return None
    p = _artifact_path(video_id)
    if not p.exists():
        return None
    try:
        art = PunctuationArtifact.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — treat any unreadable/stale cache as absent
        return None
    if (art.transcript_fingerprint != transcript_fingerprint(words, source)
            or art.token_count != len(words) or art.model != model
            or art.prompt_version != prompt_version):
        return None
    return art.result


def save_artifact(video_id: str, words: list[TimedWord], source: str, model: str,
                  prompt_version: str, result: PunctuationResult) -> None:
    if not video_id:
        return
    art = PunctuationArtifact(
        video_id=video_id, transcript_fingerprint=transcript_fingerprint(words, source),
        token_count=len(words), model=model, prompt_version=prompt_version, result=result)
    try:
        p = _artifact_path(video_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(art.model_dump_json(), encoding="utf-8")
    except Exception:  # noqa: BLE001 — cache write is best-effort
        pass


# ── per-chunk cache ──────────────────────────────────────────────────────────
def chunk_key(chunk: TranscriptChunk, words: list[TimedWord], model: str,
              prompt_version: str) -> str:
    payload = "|".join(
        f"{words[gi].id}:{words[gi].word.strip()}:{round(words[gi].start, 2)}:{words[gi].speaker or ''}"
        for gi in chunk.token_ids
    )
    return hashlib.sha1(f"{model}|{prompt_version}|{payload}".encode("utf-8")).hexdigest()


def _chunk_path(video_id: str, key: str) -> Path:
    return _video_dir(video_id) / "punctuation" / "chunks" / f"{key}.json"


def load_chunk(video_id: str, key: str,
               chunk: TranscriptChunk) -> Optional[tuple[dict[int, Annotation], str]]:
    if not video_id:
        return None
    p = _chunk_path(video_id, key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ann = {int(k): Annotation.model_validate(v) for k, v in data["ann"].items()}
        if set(ann.keys()) != set(chunk.token_ids):
            return None
        return ann, str(data.get("status", "complete"))
    except Exception:  # noqa: BLE001
        return None


def save_chunk(video_id: str, key: str, ann: dict[int, Annotation], status: str) -> None:
    if not video_id:
        return
    try:
        p = _chunk_path(video_id, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"status": status, "ann": {str(k): v.model_dump() for k, v in ann.items()}}
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
