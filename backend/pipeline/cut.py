"""Frame-accurate clip cutting with ffmpeg (re-encode, NOT stream-copy).

`-ss` before `-i` does a fast seek to the nearest prior keyframe, then the decoder
trims to the exact millisecond — fast AND accurate, so the clip lands on the period.
`-movflags +faststart` lets the <video> element start playing before the full file
downloads (matters over the LAN). Cuts run concurrently under a small semaphore.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Callable, Optional

from .. import config
from ..errors import PipelineError

ProgressCb = Optional[Callable[[float, str], None]]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:24] or "clip"


def _export_fname(n: int, facet: str, start: float, end: float) -> str:
    """Range-keyed name: different boundaries → different file (stale cache can't serve)."""
    return f"clip_{n}_{_slug(facet)}_{int(round(start * 1000))}_{int(round(end * 1000))}.mp4"


def finalize_output(tmp: Path, out: Path, rc: int, err: str) -> None:
    """Atomically promote an ffmpeg output: rename tmp→out only on success with real bytes;
    otherwise remove the partial tmp and raise. A crash/failure can never poison the cache."""
    if rc == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(out)
        return
    tmp.unlink(missing_ok=True)
    tail = err.strip().splitlines()
    raise PipelineError(f"Clip cut failed: {tail[-1] if tail else rc}")


def _venc_args(encoder: str) -> list[str]:
    """Video-encoder flags. h264_videotoolbox = Apple-Silicon hardware (rate-controlled, ~2.5×
    faster, still frame-accurate); libx264 = software (CRF). yuv420p for universal <video> support."""
    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", config.CUT_VIDEO_BITRATE, "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-crf", str(config.CRF), "-preset", config.PRESET, "-pix_fmt", "yuv420p"]


def build_cmd(src: str, start_s: float, dur_s: float, out: Path,
              encoder: str = "libx264") -> list[str]:
    return [
        config.FFMPEG_BIN, "-nostdin", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{dur_s:.3f}",
        *_venc_args(encoder),
        "-c:a", "aac", "-b:a", config.AUDIO_BITRATE,
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        str(out),
    ]


async def cut_clips(src: str, video_id: str, segments: list[dict], settings: dict,
                    progress: ProgressCb = None) -> list[dict]:
    out_dir = config.OUTPUT_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(config.CUT_CONCURRENCY)
    total = len(segments)
    done = 0
    results: list[Optional[dict]] = [None] * total

    def emit(frac: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    async def one(i: int, seg: dict) -> None:
        nonlocal done
        start = float(seg["start"])
        disp_end = float(seg["end"])                       # period boundary (for display)
        cut_end = float(seg.get("cut_end", disp_end))      # actual cut point (+tail pad)
        dur = max(0.05, round(cut_end - start, 3))
        fname = f"clip_{i + 1}_{_slug(seg.get('facet', 'clip'))}.mp4"
        out = out_dir / fname
        tmp = out.with_name(out.stem + ".tmp.mp4")

        async def _encode(encoder: str) -> tuple[int, str]:
            proc = await asyncio.create_subprocess_exec(
                *build_cmd(src, start, dur, tmp, encoder),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="ignore").strip()
                if line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1] or 0)
                    except ValueError:
                        continue
                    frac = min(1.0, (us / 1e6) / dur) if dur else 1.0
                    emit((done + frac) / total, f"Cutting clip {i + 1}/{total}")
            e = (await proc.stderr.read()).decode(errors="ignore") if proc.stderr else ""
            return await proc.wait(), e

        async with sem:
            rc, err = await _encode(config.CUT_ENCODER)
            if (rc != 0 or not tmp.exists() or tmp.stat().st_size == 0) and config.CUT_ENCODER != "libx264":
                tmp.unlink(missing_ok=True)                # hardware encode failed → software fallback
                rc, err = await _encode("libx264")

        if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            tail = err.strip().splitlines()
            raise PipelineError(f"ffmpeg failed on clip {i + 1}: {tail[-1] if tail else rc}")
        tmp.replace(out)

        done += 1
        emit(done / total, f"Cut {done}/{total} clips")
        results[i] = {
            "n": i + 1,
            "facet": seg.get("facet", "other"),
            "reason": seg.get("reason", ""),
            "start": round(start, 3),
            "end": round(disp_end, 3),
            "duration": round(disp_end - start, 3),
            "path": f"/clips/{video_id}/{fname}",
        }

    await asyncio.gather(*(one(i, s) for i, s in enumerate(segments)))
    clips = [c for c in results if c is not None]

    (out_dir / "clips.json").write_text(json.dumps(clips, indent=2), encoding="utf-8")
    return clips
