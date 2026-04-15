"""
In-process pub/sub bus for refinement-job progress events (Phase D.1).

Backend workers emit structured events via `ingestion.logging_config.publish_progress(...)`
which pushes them onto this bus keyed on the current `job_id` contextvar. The
`/api/reels/refinement-stream/{job_id}` endpoint tails the bus and emits each event
as a newline-delimited JSON line to iOS / webapp clients.

Design notes:
  - In-process only. Running multiple replicas behind a load balancer would require
    Redis pub/sub or similar; for now the FastAPI app on Railway is single-process and
    a refinement job stays on the replica that queued it. Reconnection survives brief
    hiccups because events are buffered (`BUFFER_SIZE`) and replayed from
    `cursor_ms` on subscribe.
  - Bounded memory: at most `MAX_JOBS` topics, each with a `BUFFER_SIZE` ring buffer
    of events. Idle topics evict after `TOPIC_TTL_SEC`.
  - Asyncio-friendly: subscribers get an `asyncio.Queue` that is fed by the thread-safe
    `publish()` using `loop.call_soon_threadsafe`. This lets the worker thread (which
    runs `_run_refinement_job`) publish without blocking on an async primitive.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

BUFFER_SIZE: int = 256
"""Max events retained per job_id for replay on late subscribers."""

MAX_JOBS: int = 512
"""Hard cap on tracked topics. Eviction is LRU on last-activity-ts."""

TOPIC_TTL_SEC: float = 15 * 60.0
"""Topics with no activity for this long are dropped."""

SUBSCRIBER_QUEUE_MAXSIZE: int = 128
"""Per-subscriber queue bound. Overflow drops oldest to prevent back-pressure."""


# --------------------------------------------------------------------------- #
# Internal state
# --------------------------------------------------------------------------- #


@dataclass
class _Topic:
    buffer: deque[tuple[int, dict[str, Any]]] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    subscribers: list["_Subscriber"] = field(default_factory=list)
    terminated: bool = False
    last_activity_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[dict[str, Any] | None]


_topics: dict[str, _Topic] = {}
_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _evict_idle_topics_locked() -> None:
    """Drop topics inactive for longer than the TTL, or oldest-first if over MAX_JOBS."""
    now = _now_ms()
    stale = [
        jid
        for jid, topic in _topics.items()
        if (now - topic.last_activity_ms) > (TOPIC_TTL_SEC * 1000)
        and not topic.subscribers
    ]
    for jid in stale:
        _topics.pop(jid, None)

    if len(_topics) > MAX_JOBS:
        # LRU on last_activity_ms. Skip topics with live subscribers.
        victims = sorted(
            (jid for jid, t in _topics.items() if not t.subscribers),
            key=lambda jid: _topics[jid].last_activity_ms,
        )
        for jid in victims[: len(_topics) - MAX_JOBS]:
            _topics.pop(jid, None)


def _deliver_threadsafe(sub: _Subscriber, event: dict[str, Any] | None) -> None:
    """Put an event on a subscriber's queue from an arbitrary thread."""

    def _put() -> None:
        if sub.queue.full():
            # Drop the oldest to keep the subscriber live rather than blocking.
            try:
                sub.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        sub.queue.put_nowait(event)

    try:
        sub.loop.call_soon_threadsafe(_put)
    except RuntimeError:
        # Event loop closed; subscriber is gone.
        pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def publish(job_id: str, event: dict[str, Any]) -> None:
    """
    Publish an event to all subscribers of `job_id` and append it to the replay buffer.

    Thread-safe. Non-blocking. Safe to call before any subscriber attaches (events
    are buffered up to `BUFFER_SIZE`).
    """
    if not job_id or job_id == "-":
        return
    enriched = dict(event)
    enriched.setdefault("ts_ms", _now_ms())
    seq_ms = int(enriched["ts_ms"])

    with _lock:
        topic = _topics.get(job_id)
        if topic is None:
            _evict_idle_topics_locked()
            topic = _Topic()
            _topics[job_id] = topic
        topic.buffer.append((seq_ms, enriched))
        topic.last_activity_ms = seq_ms
        subs = list(topic.subscribers)

    for sub in subs:
        _deliver_threadsafe(sub, enriched)


def terminate(job_id: str, final_event: dict[str, Any] | None = None) -> None:
    """
    Mark a job's stream complete. Subscribers still attached will receive the optional
    `final_event` followed by a `None` sentinel, which `subscribe()` turns into
    iterator exit.
    """
    if not job_id or job_id == "-":
        return
    with _lock:
        topic = _topics.get(job_id)
        if topic is None:
            return
        topic.terminated = True
        topic.last_activity_ms = _now_ms()
        subs = list(topic.subscribers)

    if final_event is not None:
        publish(job_id, final_event)

    for sub in subs:
        _deliver_threadsafe(sub, None)


async def subscribe(
    job_id: str,
    *,
    replay_from_ms: int = 0,
    idle_timeout_sec: float = 60.0,
) -> AsyncIterator[dict[str, Any]]:
    """
    Async iterator over events for `job_id`.

    - Replays any buffered events whose `ts_ms > replay_from_ms` before streaming live.
    - Exits cleanly when the topic is terminated (see `terminate()`).
    - Exits on `idle_timeout_sec` with no events (prevents dangling subscribers).
    """
    if not job_id or job_id == "-":
        return

    loop = asyncio.get_running_loop()
    sub = _Subscriber(loop=loop, queue=asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_MAXSIZE))

    with _lock:
        topic = _topics.get(job_id)
        if topic is None:
            _evict_idle_topics_locked()
            topic = _Topic()
            _topics[job_id] = topic
        topic.subscribers.append(sub)
        replay = [evt for (ts_ms, evt) in topic.buffer if ts_ms > replay_from_ms]
        terminated_already = topic.terminated

    try:
        for evt in replay:
            yield evt
        if terminated_already:
            return

        while True:
            try:
                evt = await asyncio.wait_for(sub.queue.get(), timeout=idle_timeout_sec)
            except asyncio.TimeoutError:
                return
            if evt is None:
                return
            yield evt
    finally:
        with _lock:
            topic = _topics.get(job_id)
            if topic is not None:
                try:
                    topic.subscribers.remove(sub)
                except ValueError:
                    pass


def snapshot(job_id: str) -> list[dict[str, Any]]:
    """Return a copy of the current replay buffer (for debugging / test assertions)."""
    with _lock:
        topic = _topics.get(job_id)
        if topic is None:
            return []
        return [evt for (_ts, evt) in topic.buffer]


def active_topics() -> list[str]:
    """List currently-tracked job ids (debug/metrics)."""
    with _lock:
        return list(_topics.keys())


__all__ = [
    "publish",
    "subscribe",
    "terminate",
    "snapshot",
    "active_topics",
    "BUFFER_SIZE",
    "MAX_JOBS",
    "TOPIC_TTL_SEC",
]
