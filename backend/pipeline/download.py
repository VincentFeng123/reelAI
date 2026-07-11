"""Download a YouTube video (720p mp4) + extract a 16 kHz mono audio track for Groq.

One network download (the merged video), then a local ffmpeg pass to derive the
audio — this guarantees audio and video share the exact same timeline, which the
cut-precision depends on. Cached by video_id under work/<video_id>/.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

from .. import config
from ..errors import PipelineError

ProgressCb = Optional[Callable[[float, str], None]]

_ID_PATTERNS = [
    re.compile(r"(?:v=|/shorts/|/embed/|/live/|youtu\.be/)([A-Za-z0-9_-]{11})"),
]


def extract_video_id(url: str) -> Optional[str]:
    for pat in _ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


# yt-dlp info-dict keys that feed genre detection (GEN1). categories (e.g. ['Comedy']) is the
# load-bearing non-lecture signal; artist/track/genre are often NULL for user parodies → we keep
# them but treat null as no-signal downstream (never force music on a null artist).
def _meta_keys(info: dict) -> dict:
    info = info or {}
    return {
        "categories": list(info.get("categories") or []),
        "tags": list(info.get("tags") or []),
        "artist": info.get("artist"),
        "track": info.get("track"),
        "genre": info.get("genre"),
    }


def probe_metadata(url: str) -> dict:
    """Cheap metadata-only probe (no media download) so the genre-metadata tier also works in
    supadata/fast mode, which never runs a full download(). Fail-soft: any error → {}.

    Returns the same keys as download()'s meta ({categories, tags, artist, track, genre}); an
    empty dict means "no signal", which the detector treats as a no-op.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:  # noqa: BLE001 — metadata is best-effort; never break the pipeline
        return {}
    if not isinstance(info, dict):
        return {}
    return _meta_keys(info)


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    cmd = [
        config.FFMPEG_BIN, "-nostdin", "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", str(config.TARGET_AUDIO_SR),
        "-ac", str(config.TARGET_AUDIO_CH),
        "-c:a", "aac", "-b:a", f"{config.AUDIO_BITRATE_K}k",
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        raise PipelineError(f"ffmpeg audio extraction failed: {tail[-1] if tail else proc.returncode}")


def download(url: str, settings: dict, progress: ProgressCb = None) -> dict:
    """Return {video_id, video_path, audio_path, title, duration}."""
    def emit(frac: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    video_id = extract_video_id(url)
    # Build the workdir lazily; if we can't parse the id, yt-dlp resolves it below.
    workdir = config.WORK_DIR / video_id if video_id else None
    if workdir:
        workdir.mkdir(parents=True, exist_ok=True)
        video_path = workdir / "video.mp4"
        audio_path = workdir / "audio.m4a"
        if video_path.exists() and audio_path.exists():
            emit(1.0, "Using cached download")
            return {
                "video_id": video_id,
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "title": _read_title(workdir),
                "duration": None,
                **_read_meta(workdir),   # GEN1: cached yt-dlp metadata (categories/tags/…)
            }

    max_res = int(settings.get("max_resolution", config.MAX_RESOLUTION))
    # Prefer ≤max_res video+audio; fall back progressively so a match always exists.
    fmt = (
        f"bestvideo[height<={max_res}]+bestaudio/"
        f"best[height<={max_res}]/"
        f"bestvideo+bestaudio/best"
    )

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            frac = (done / total) if total else 0.0
            emit(frac * 0.95, "Downloading video…")
        elif d.get("status") == "finished":
            emit(0.97, "Merging…")

    # Temporary template until we know the id (yt-dlp fills %(id)s).
    tmp_tmpl = str((workdir or config.WORK_DIR) / ("video.%(ext)s" if workdir else "%(id)s.video.%(ext)s"))
    ydl_opts = {
        "format": fmt,
        "outtmpl": tmp_tmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "ffmpeg_location": str(Path(config.FFMPEG_BIN).parent),
        "retries": 5,
        "fragment_retries": 5,
        # Let yt-dlp fetch the EJS solver so deno can solve YouTube's n-challenge
        # (otherwise video formats may be missing or downloads get throttled).
        "remote_components": ["ejs:github"],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise PipelineError(_classify_download_error(str(e)))
    except Exception as e:  # noqa: BLE001
        raise PipelineError(f"Download failed: {e}")

    rid = info.get("id")
    if not rid:
        raise PipelineError("Could not resolve the video id.")
    if not workdir:
        # Re-home files under work/<id>/ now that we know the id.
        workdir = config.WORK_DIR / rid
        workdir.mkdir(parents=True, exist_ok=True)
        src = config.WORK_DIR / f"{rid}.video.mp4"
        video_path = workdir / "video.mp4"
        if src.exists():
            src.replace(video_path)
        audio_path = workdir / "audio.m4a"
    video_id = rid

    if not video_path.exists():
        # Fallback: yt-dlp may have used a different ext; find the merged file.
        cands = sorted(workdir.glob("video.*")) or sorted(config.WORK_DIR.glob(f"{rid}.video.*"))
        if not cands:
            raise PipelineError("Downloaded file not found after merge.")
        cands[0].replace(video_path)

    emit(0.98, "Extracting audio…")
    _extract_audio(video_path, audio_path)
    _write_title(workdir, info.get("title", ""))
    meta = _meta_keys(info)                       # GEN1: surface genre-detection metadata
    _write_meta(workdir, meta)                    # cache it so a later cache-hit keeps the signal
    emit(1.0, "Download complete")

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "title": info.get("title", ""),
        "duration": info.get("duration"),
        **meta,
    }


def _write_title(workdir: Path, title: str) -> None:
    try:
        (workdir / "title.txt").write_text(title, encoding="utf-8")
    except OSError:
        pass


def _read_title(workdir: Path) -> str:
    p = workdir / "title.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _write_meta(workdir: Path, meta: dict) -> None:
    try:
        (workdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    except (OSError, TypeError):
        pass


def _read_meta(workdir: Path) -> dict:
    p = workdir / "meta.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return _meta_keys(data) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _classify_download_error(msg: str) -> str:
    s = msg.lower()
    if "private video" in s:
        return "This video is private and can't be downloaded."
    if "age" in s and ("restrict" in s or "confirm" in s):
        return "This video is age-restricted and can't be fetched."
    if "members-only" in s or "join this channel" in s:
        return "This is a members-only video."
    if "not available in your country" in s or "blocked it in your country" in s or "geo" in s:
        return "This video is region-locked."
    if "video unavailable" in s or "removed" in s or "no longer available" in s:
        return "Video unavailable (removed, or the URL is wrong)."
    if "sign in" in s and "bot" in s:
        return "YouTube is asking for sign-in/bot verification for this video."
    return f"Download failed: {msg.strip().splitlines()[-1] if msg.strip() else msg}"
