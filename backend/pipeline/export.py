"""On-demand high-res export of a single clip.

YouTube throttles DASH range/section downloads heavily, so instead we download the
full video once at the requested resolution (fast, unthrottled, cached per
resolution) and cut the clip locally with ffmpeg — reliable and frame-accurate.
Subsequent exports of the same video reuse the cached source.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import yt_dlp

from .. import config
from ..errors import PipelineError
from .cut import build_cmd, finalize_output, _export_fname


def _download_full(url: str, res: int, out_tmpl: str) -> None:
    opts = {
        "format": (
            f"bestvideo[height<={res}]+bestaudio/best[height<={res}]/bestvideo+bestaudio/best"
        ),
        "outtmpl": out_tmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": str(Path(config.FFMPEG_BIN).parent),
        "remote_components": ["ejs:github"],
        "retries": 5,
        "fragment_retries": 5,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise PipelineError(f"Export download failed: {str(e).strip().splitlines()[-1]}")


def _ensure_source(url: str, video_id: str, res: int) -> Path:
    work = config.WORK_DIR / video_id
    work.mkdir(parents=True, exist_ok=True)
    # Reuse any already-downloaded full video (from a prior export or whisper run).
    for existing in (work / f"source_{res}.mp4", work / "video.mp4"):
        if existing.exists():
            return existing
    _download_full(url, res, str(work / f"source_{res}.%(ext)s"))
    src = work / f"source_{res}.mp4"
    if not src.exists():
        cands = [c for c in work.glob(f"source_{res}.*") if c.suffix.lower() in (".mp4", ".mkv", ".webm")]
        if not cands:
            raise PipelineError("Export produced no source file.")
        cands[0].replace(src)
    return src


async def export_clip(
    url: str, video_id: str, n: int, facet: str, start: float, end: float, res: int
) -> dict:
    out_dir = config.OUTPUT_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _export_fname(n, facet, float(start), float(end))
    out = out_dir / fname
    served = f"/clips/{video_id}/{fname}"
    if out.exists() and out.stat().st_size > 0:
        return {"path": served}

    src = await asyncio.to_thread(_ensure_source, url, video_id, int(res))

    dur = max(0.1, round(float(end) - float(start), 3))
    tmp = out.with_name(out.stem + ".tmp.mp4")
    proc = await asyncio.create_subprocess_exec(
        *build_cmd(str(src), float(start), dur, tmp),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    err = (await proc.stderr.read()).decode(errors="ignore") if proc.stderr else ""
    rc = await proc.wait()
    finalize_output(tmp, out, rc, err)
    return {"path": served}
