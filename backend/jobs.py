"""In-memory job registry with thread-safe SSE fan-out.

Pipeline stages run in worker threads, so their progress callbacks fire off the
event loop. Every Job mutation is funneled through loop.call_soon_threadsafe, so
all Job field writes happen on the loop thread and no locks are needed. Each SSE
client gets its own bounded queue and an immediate snapshot replay on subscribe.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ProgressEvent:
    stage: str
    pct: float
    message: str


@dataclass
class Job:
    id: str
    url: str
    topic: str
    settings: dict
    status: Status = Status.PENDING
    current_stage: str = "downloading"
    pct: float = 0.0
    message: str = "Queued"
    video_id: Optional[str] = None
    title: str = ""
    clips: list[dict] = field(default_factory=list)
    notes: str = ""
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    _subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    def snapshot(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "stage": self.current_stage,
            "pct": round(self.pct, 1),
            "message": self.message,
            "title": self.title,
            "video_id": self.video_id,
            "clips": self.clips,
            "notes": self.notes,
            "error": self.error,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def create(self, url: str, topic: str, settings: dict) -> Job:
        job = Job(id=uuid.uuid4().hex, url=url, topic=topic, settings=settings)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    # ── SSE plumbing ──────────────────────────────────────────────────────
    @staticmethod
    def _envelope(job: Job) -> dict:
        if job.status == Status.DONE:
            return {"event": "done", "data": job.snapshot()}
        if job.status in (Status.ERROR, Status.CANCELLED):
            return {"event": "failed", "data": job.snapshot()}
        return {"event": "progress", "data": job.snapshot()}

    def subscribe(self, job: Job) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        job._subscribers.add(q)
        q.put_nowait(self._envelope(job))  # replay current state for late/reconnecting clients
        return q

    def unsubscribe(self, job: Job, q: asyncio.Queue) -> None:
        job._subscribers.discard(q)

    def _fanout(self, job: Job) -> None:
        env = self._envelope(job)
        for q in list(job._subscribers):
            try:
                q.put_nowait(env)
            except asyncio.QueueFull:
                pass

    def _on_loop(self, fn) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(fn)
        else:
            fn()

    # ── mutations (safe from any thread) ──────────────────────────────────
    def publish(self, job: Job, ev: ProgressEvent) -> None:
        def _apply():
            job.status = Status.RUNNING
            job.current_stage, job.pct, job.message = ev.stage, ev.pct, ev.message
            self._fanout(job)
        self._on_loop(_apply)

    def set_meta(self, job: Job, *, video_id: Optional[str] = None, title: Optional[str] = None) -> None:
        def _apply():
            if video_id is not None:
                job.video_id = video_id
            if title is not None:
                job.title = title
        self._on_loop(_apply)

    def finish(self, job: Job, clips: list[dict], notes: str = "") -> None:
        def _apply():
            job.status = Status.DONE
            job.current_stage, job.pct, job.message = "done", 100.0, "Done"
            job.clips = clips
            job.notes = notes
            self._fanout(job)
        self._on_loop(_apply)

    def fail(self, job: Job, message: str, notes: str = "") -> None:
        def _apply():
            job.status = Status.ERROR
            job.error = message
            job.message = message
            job.notes = notes
            self._fanout(job)
        self._on_loop(_apply)


registry = JobRegistry()
