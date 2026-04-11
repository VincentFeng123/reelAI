"""
Structured logging for the ingestion pipeline.

Every log line emitted from ingestion modules is tagged with a trace_id (set via a contextvar
at the start of each `IngestionPipeline.ingest_url` call) so the full request journey can be
grepped in Railway logs. Also exposes a `log_event(...)` helper that emits structured JSON
payloads suitable for DMCA audit trail.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from typing import Any

_trace_id_var: ContextVar[str] = ContextVar("ingest_trace_id", default="-")


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

    The JSON payload always includes `trace_id` (auto-injected from the contextvar).
    Use this for ingest lifecycle events — especially `ingest_completed`, which must
    be searchable by `source_url` for DMCA takedowns.
    """
    payload = {"trace_id": _trace_id_var.get(), **fields}
    try:
        rendered = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(payload)
    logger.log(level, "%s %s", event, rendered)


__all__ = [
    "get_ingest_logger",
    "new_trace_id",
    "set_trace_id",
    "current_trace_id",
    "log_event",
]
