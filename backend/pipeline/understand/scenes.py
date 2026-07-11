"""Scene-cut detection + keyframe extraction via ffmpeg (zero new deps).

The visual backbone is a uniform keyframe grid (every ``KEYFRAME_GRID_S`` seconds); ffmpeg's
``select='gt(scene,…)'`` filter adds extra samples at hard cuts so slide/board changes aren't
missed between grid points. Frames are extracted to ``work/<id>/keyframes/`` and de-duplicated
by perceptual dHash so a static board doesn't spend vision budget on identical frames. Each
kept keyframe becomes a ``Scene`` spanning until the next kept keyframe.

Everything here degrades to an empty list on any failure (missing ffmpeg, bad video, timeout);
the caller records ``"scenes"`` in ``Perception.degraded`` and the pipeline stays transcript-only.
"""
from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from ... import config
from .models import Scene

ProgressCb = Optional[Callable[[float, str], None]]


def _scene_times(video_path: str, duration: float) -> list[float]:
    """Timestamps of frames whose scene score exceeds ``SCENE_THRESHOLD``.

    Downscales to 320px wide before scoring so the full-decode pass is cheap on long videos.
    Times out (→ empty, grid-only) rather than hanging the job on a pathological input.
    """
    cmd = [
        config.FFMPEG_BIN, "-nostdin", "-hide_banner", "-an", "-i", video_path,
        "-filter:v", f"scale=320:-2,select='gt(scene,{config.SCENE_THRESHOLD})',showinfo",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=max(180.0, duration))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr or "")]


def _uniform_grid(duration: float, step: float) -> list[float]:
    t, out = 0.0, []
    while t < duration:
        out.append(round(t, 2))
        t += step
    return out


def _dhash(path: str) -> Optional[int]:
    """9×8 → 8×8 difference hash of a keyframe (perceptual, for near-dup detection)."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("L").resize((9, 8))
        a = np.asarray(img, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _extract_frame(video_path: str, t: float, out: Path) -> bool:
    """Seek-then-grab one JPEG at ``t`` (fast pre-input seek; accuracy is fine for captioning)."""
    try:
        subprocess.run(
            [config.FFMPEG_BIN, "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "3", str(out)],
            capture_output=True, timeout=60.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return out.exists() and out.stat().st_size > 0


def detect_and_extract(video_path: str, video_id: str, duration: float,
                       progress: ProgressCb = None) -> list[Scene]:
    """Extract de-duplicated keyframes → contiguous ``Scene`` spans (or [] on failure)."""
    if duration <= 0 or not Path(video_path).exists():
        return []

    # candidate times: uniform grid ∪ scene cuts, min-gap enforced, count-capped. The scene-cut
    # pass is a full decode (the slow part) — SCENE_DETECTION=0 drops it and keeps grid-only frames.
    scene_cuts = set(_scene_times(video_path, duration)) if config.SCENE_DETECTION else set()
    times = sorted(set(_uniform_grid(duration, config.KEYFRAME_GRID_S)) | scene_cuts)
    kept_times: list[float] = []
    for t in times:
        if t < 0 or t >= duration:
            continue
        if not kept_times or (t - kept_times[-1]) >= config.KEYFRAME_MIN_GAP_S:
            kept_times.append(t)
    # cap the count by subsampling EVENLY across the whole video — head-truncation would leave the
    # tail of a long lecture unanalyzed.
    if len(kept_times) > config.KEYFRAME_MAX:
        step = len(kept_times) / config.KEYFRAME_MAX
        kept_times = [kept_times[int(i * step)] for i in range(config.KEYFRAME_MAX)]
    if not kept_times:
        return []

    kf_dir = config.WORK_DIR / video_id / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    for old in kf_dir.glob("kf_*.jpg"):                    # drop stale frames from a prior run
        old.unlink(missing_ok=True)

    # Latency lever: each keyframe is an INDEPENDENT `ffmpeg -ss` subprocess, so extract them all
    # concurrently (config.VISION_WORKERS; =1 == the exact serial path). Every ordering-sensitive
    # step stays SERIAL: the dHash dedup below runs over the finished JPEGs IN TIME ORDER (the
    # kept_times index order), so the scenes list, indexes, spans, kept files and unlinked dups are
    # byte-identical to serial regardless of which extraction finishes first.
    ok: list[bool] = [False] * len(kept_times)
    workers = max(1, min(int(config.VISION_WORKERS), len(kept_times)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_extract_frame, video_path, t, kf_dir / f"kf_{i:04d}.jpg"): i
                for i, t in enumerate(kept_times)}
        for fut in as_completed(futs):
            ok[futs[fut]] = fut.result()

    scenes: list[Scene] = []
    last_hash: Optional[int] = None
    total = len(kept_times)
    for i, t in enumerate(kept_times):
        out = kf_dir / f"kf_{i:04d}.jpg"
        if not ok[i]:
            continue
        h = _dhash(str(out))
        if last_hash is not None and h is not None and _hamming(h, last_hash) <= config.DHASH_HAMMING_DROP:
            out.unlink(missing_ok=True)                    # near-identical to previous kept → drop
            if progress:
                progress((i + 1) / total, f"Keyframes {i + 1}/{total}")
            continue
        if h is not None:
            last_hash = h
        scenes.append(Scene(index=len(scenes), start=t, end=duration,
                            keyframe_time=t, keyframe_path=str(out)))
        if progress:
            progress((i + 1) / total, f"Keyframes {i + 1}/{total}")

    # make spans contiguous: each kept keyframe holds until the next kept one (so a slide that
    # de-duped away several frames still spans its full on-screen lifetime).
    for k in range(len(scenes) - 1):
        scenes[k].end = scenes[k + 1].start
    return scenes
