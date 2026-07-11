"""FastAPI app: job creation, SSE progress stream, clip serving, zip download,
and same-origin serving of the built React SPA.
"""
from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from . import config
from .errors import friendly_error
from .jobs import Status, registry
from .orchestrator import run_pipeline
from .schemas import CreateJobReq, CreateJobResp

EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipe")


def _zipable_files(clips: list[dict], folder: Path) -> list[Path]:
    """Existing rendered files for a job — embed-mode (path=None) and missing files skipped."""
    out: list[Path] = []
    for c in clips or []:
        p = c.get("path")
        if not p:
            continue
        fp = folder / Path(p).name
        if fp.exists():
            out.append(fp)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.bind_loop(asyncio.get_running_loop())
    yield
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="YouTube Topic Clipper", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.DEV_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    from .pipeline.understand import diarize, ocr, vision
    return {
        "ok": True,
        "transcriber": config.TRANSCRIBER,
        "llm_provider": config.LLM_PROVIDER,
        "supadata_key": bool(config.SUPADATA_API_KEY),
        "gemini_key": bool(config.GEMINI_API_KEY),
        "groq_key": bool(config.GROQ_API_KEY),
        "gemini_model": config.GEMINI_MODEL,
        # perception capabilities (degrade individually when unavailable)
        "perception": {
            "multimodal": bool(config.MULTIMODAL),
            "vision": vision.available(),                  # Gemini keyframe captioning
            "ocr": ocr.available(),                        # optional local OCR
            "diarization": diarize.available(),            # pyannote speaker turns (needs HF token)
        },
    }


@app.post("/jobs", response_model=CreateJobResp, status_code=202)
async def create_job(req: CreateJobReq):
    settings = {**config.DEFAULTS, **(req.settings or {})}
    job = registry.create(req.url, req.topic, settings)
    job.status = Status.RUNNING
    job._task = asyncio.create_task(run_pipeline(job, EXECUTOR))
    return CreateJobResp(job_id=job.id)


@app.get("/jobs/{job_id}/stream")
async def stream(job_id: str, request: Request):
    job = registry.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")

    async def gen():
        q = registry.subscribe(job)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    env = await asyncio.wait_for(q.get(), timeout=config.SSE_HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {"event": env["event"], "data": json.dumps(env["data"])}
                if env["event"] in ("done", "failed"):
                    break
        finally:
            registry.unsubscribe(job, q)

    return EventSourceResponse(gen())


@app.get("/jobs/{job_id}/zip")
def download_zip(job_id: str):
    job = registry.get(job_id)
    if not job or job.status != Status.DONE or not job.video_id:
        raise HTTPException(404, "No finished job to zip")
    folder = config.OUTPUT_DIR / job.video_id
    files = _zipable_files(job.clips, folder)
    has_manifest = (folder / "clips.json").exists()
    if not files and not has_manifest:
        raise HTTPException(409, "No rendered clips to zip — use output_mode=cut or export clips first")

    def gen():
        from zipstream import ZipStream
        zs = ZipStream(sized=False)
        for fp in files:
            zs.add_path(fp, fp.name)
        if has_manifest:
            zs.add_path(folder / "clips.json", "clips.json")
        yield from zs

    return StreamingResponse(
        gen(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="clips_{job.video_id}.zip"'},
    )


@app.post("/jobs/{job_id}/clips/{n}/export")
async def export_clip_endpoint(job_id: str, n: int):
    job = registry.get(job_id)
    if not job or job.status != Status.DONE:
        raise HTTPException(404, "No finished job")
    clip = next((c for c in job.clips if c.get("n") == n), None)
    if not clip or not clip.get("video_id"):
        raise HTTPException(404, "Clip not found")
    from .pipeline.export import export_clip

    res = int(job.settings.get("export_resolution", config.EXPORT_RESOLUTION))
    try:
        out = await export_clip(
            job.url, clip["video_id"], n, clip.get("facet", "clip"),
            clip["start"], clip["end"], res,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, friendly_error(e))
    clip["path"] = out["path"]
    return out


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = registry.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job.snapshot()


@app.get("/clips/{video_id}/{file}")
def get_clip(video_id: str, file: str, download: bool = False):
    base = (config.OUTPUT_DIR / video_id).resolve()
    safe = (base / file).resolve()
    if safe.parent != base or not safe.is_file():
        raise HTTPException(404, "Clip not found")
    headers = {"Accept-Ranges": "bytes"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{file}"'
    return FileResponse(safe, media_type="video/mp4", headers=headers)


# ── human labeling (judge calibration; page at /labeling/index.html) ─────────
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")   # YouTube ids; also blocks path tricks


class HumanClipLabel(BaseModel):
    start: float
    end: float
    understandable: bool
    failure_kinds: list[str] = Field(default_factory=list)
    needed_first: str = ""
    labeled_at: str = ""          # blank → server stamps it


class LabelsReq(BaseModel):
    video_id: str
    labels: list[HumanClipLabel] = Field(default_factory=list)
    video_note: str = ""


@app.get("/api/labels/{video_id}")
def get_labels(video_id: str):
    """Existing human label block for a video — the labeling page's resume source.
    Always {clips: [...], video_note: str} (empty block when nothing is labeled yet)."""
    from .eval.golden import human_block, load_golden
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(400, "Invalid video id")
    return human_block(load_golden(video_id))


@app.post("/api/labels")
def post_labels(req: LabelsReq):
    """Merge human labels into eval/golden/<video_id>.json under the 'human' key. Never
    clobbers hand-authored gold keys (same contract as make_golden's chapter merge);
    re-labeled clips upsert by span match within golden.HUMAN_MATCH_TOL_S."""
    from datetime import datetime

    from .eval.golden import merge_human_into_golden
    if not _VIDEO_ID_RE.match(req.video_id):
        raise HTTPException(400, "Invalid video id")
    try:
        human = merge_human_into_golden(
            req.video_id, [lab.model_dump() for lab in req.labels], req.video_note,
            labeled_at=datetime.now().isoformat(timespec="seconds"))
    except ValueError as e:                    # present-but-corrupt golden file: never clobber
        raise HTTPException(409, str(e))
    return {"ok": True, "video_id": req.video_id, "n_clips": len(human["clips"])}


# ── SPA static serving (registered LAST so API routes win) ───────────────────
if (config.STATIC_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=config.STATIC_DIR / "assets"), name="assets")


@app.get("/")
@app.get("/{full_path:path}")
async def spa(full_path: str = ""):
    index = config.STATIC_DIR / "index.html"
    cand = config.STATIC_DIR / full_path
    if full_path and cand.is_file():
        return FileResponse(cand)
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(
        {"detail": "Frontend not built. Run: cd frontend && npm install && npm run build"},
        status_code=200,
    )
