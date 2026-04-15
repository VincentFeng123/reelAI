"""
Structured logging for the ingestion pipeline.

Every log line emitted from ingestion modules is tagged with a trace_id (set via a contextvar
at the start of each `IngestionPipeline.ingest_url` call) so the full request journey can be
grepped in Railway logs. Also exposes a `log_event(...)` helper that emits structured JSON
payloads suitable for DMCA audit trail.

Phase D.2 additions:
  - `@instrumented(stage=...)` decorator that wraps sync or async callables with
    stage_start / stage_end structured events (timing, outcome, error_class).
  - A `job_id` contextvar so backend refinement worker events can be correlated to
    the `/api/reels/refinement-stream/{job_id}` NDJSON progress endpoint (Phase D.1).
  - `publish_progress(...)` shim that both logs the event AND pushes it onto the
    in-process progress bus so streaming endpoints can tail it.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Optional, TypeVar

_trace_id_var: ContextVar[str] = ContextVar("ingest_trace_id", default="-")
_job_id_var: ContextVar[str] = ContextVar("ingest_job_id", default="-")


class _TraceIdFilter(logging.Filter):
    """Injects the current trace_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get()
        return True


_filter_installed: set[str] = set()


def get_ingest_logger(name: str) -> logging.Logger:
    """
    Return a logger with the trace_id filter attached.

    We install the filter idempotently on each logger we touch so that any handler inherited
    from the root logger sees the trace_id attribute without us having to reach into the root
    logger's configuration.
    """
    logger = logging.getLogger(name)
    if name not in _filter_installed:
        logger.addFilter(_TraceIdFilter())
        _filter_installed.add(name)
    return logger


def new_trace_id() -> str:
    """Generate a fresh trace id. Callers store it via `set_trace_id()`."""
    return uuid.uuid4().hex[:16]


def set_trace_id(trace_id: str | None) -> str:
    """
    Set the current trace id for subsequent log calls in this task.

    If `trace_id` is falsy, generates a new one. Returns the effective trace id.
    """
    effective = trace_id or new_trace_id()
    _trace_id_var.set(effective)
    return effective


def current_trace_id() -> str:
    return _trace_id_var.get()


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """
    Emit a structured log line: `<event> {json payload}`.

    The JSON payload always includes `trace_id` and `job_id` (auto-injected from the
    contextvars). Use this for ingest lifecycle events — especially `ingest_completed`,
    which must be searchable by `source_url` for DMCA takedowns.
    """
    payload = {"trace_id": _trace_id_var.get(), "job_id": _job_id_var.get(), **fields}
    try:
        rendered = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(payload)
    logger.log(level, "%s %s", event, rendered)


def set_job_id(job_id: str | None) -> str:
    """Set the current refinement job id for subsequent log calls in this task."""
    effective = job_id or "-"
    _job_id_var.set(effective)
    return effective


def current_job_id() -> str:
    return _job_id_var.get()


# --------------------------------------------------------------------------- #
# @instrumented decorator
# --------------------------------------------------------------------------- #

T = TypeVar("T")


def _classify_outcome(exc: BaseException | None) -> str:
    if exc is None:
        return "ok"
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    return "error"


def instrumented(
    stage: str,
    *,
    logger_name: str = "reelai.ingest",
    level: int = logging.INFO,
    publish: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Wrap a function so it emits `stage_start` and `stage_end` structured events.

    The decorator works for sync and async callables. It records wall-clock latency
    (duration_ms) and outcome (`ok` / `error` / `cancelled`) and, on error, the
    exception class name (`error_class`).

    When `publish=True` (default) the event is also pushed onto the in-process
    progress bus via `publish_progress`, keyed on the current `job_id` contextvar.
    Sync callers from a hot loop can set `publish=False` to avoid the dispatch cost.

    Usage:
        @instrumented("resolve-candidate")
        async def resolve_candidate(video_id: str) -> dict: ...
    """
    logger = get_ingest_logger(logger_name)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        is_coro = inspect.iscoroutinefunction(func)

        if is_coro:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                _emit(logger, level, "stage_start", stage=stage, publish=publish)
                exc: BaseException | None = None
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except BaseException as e:
                    exc = e
                    raise
                finally:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    _emit(
                        logger,
                        level if exc is None else logging.WARNING,
                        "stage_end",
                        stage=stage,
                        duration_ms=duration_ms,
                        outcome=_classify_outcome(exc),
                        error_class=type(exc).__name__ if exc else None,
                        publish=publish,
                    )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            _emit(logger, level, "stage_start", stage=stage, publish=publish)
            exc: BaseException | None = None
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                exc = e
                raise
            finally:
                duration_ms = int((time.monotonic() - start) * 1000)
                _emit(
                    logger,
                    level if exc is None else logging.WARNING,
                    "stage_end",
                    stage=stage,
                    duration_ms=duration_ms,
                    outcome=_classify_outcome(exc),
                    error_class=type(exc).__name__ if exc else None,
                    publish=publish,
                )

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def publish_progress(event: str, **fields: Any) -> None:
    """
    Log an event AND push it onto the in-process progress bus (Phase D.1).

    The bus is lazily imported to avoid a circular dependency at module-load time
    (services.progress_bus imports nothing from ingestion, so this is safe).
    Callers do not need to check whether a subscriber is attached — the bus
    silently buffers with TTL eviction.
    """
    logger = get_ingest_logger("reelai.progress")
    _emit(logger, logging.INFO, event, publish=True, **fields)


def _emit(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    publish: bool,
    **fields: Any,
) -> None:
    """Internal emit shim that logs and optionally publishes to the bus."""
    log_event(logger, level, event, **fields)
    if not publish:
        return
    job_id = _job_id_var.get()
    if job_id == "-" or not job_id:
        return
    # Lazy import to sidestep circular init order (services layer imports ingestion logging)
    try:
        from ..services.progress_bus import publish as _bus_publish
    except Exception:  # pragma: no cover - bus module optional at import time
        return
    _bus_publish(
        job_id,
        {
            "event": event,
            "trace_id": _trace_id_var.get(),
            "ts_ms": int(time.time() * 1000),
            **fields,
        },
    )


__all__ = [
    "get_ingest_logger",
    "new_trace_id",
    "set_trace_id",
    "current_trace_id",
    "set_job_id",
    "current_job_id",
    "log_event",
    "instrumented",
    "publish_progress",
]
