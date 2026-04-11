"""
Temp workspace lifecycle + orphan sweeper.

Everything the ingestion pipeline downloads lives inside a `TempWorkspace` context manager.
The workspace is a `tempfile.mkdtemp(prefix="reelai-ingest-")` directory that is removed
UNCONDITIONALLY on context exit — success, exception, or process signal. `shutil.rmtree`
is called with `ignore_errors=True` so a locked file can't crash cleanup.

On module import we run `sweep_orphans(max_age_sec=3600)` once to clean up any workspaces
left behind by a killed worker in a previous run. This is the safety net — don't rely on
try/finally alone.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator

from .logging_config import get_ingest_logger, log_event

logger: logging.Logger = get_ingest_logger(__name__)

_WORKSPACE_PREFIX = "reelai-ingest-"
_ORPHAN_MAX_AGE_SEC_DEFAULT = 3600  # 1 hour


@contextlib.contextmanager
def TempWorkspace(prefix: str = _WORKSPACE_PREFIX) -> Iterator[Path]:
    """
    Create a fresh temp directory, yield its Path, and delete it on exit.

    Uses the system default temp dir via `tempfile.mkdtemp` — honors `TMPDIR`. On Railway
    this resolves to `/tmp`; on Vercel it's also `/tmp` (the only writable path, but we
    refuse to run there anyway due to SERVERLESS_MODE).
    """
    workspace = Path(tempfile.mkdtemp(prefix=prefix))
    logger.debug("temp workspace created: %s", workspace)
    try:
        yield workspace
    finally:
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            # rmtree with ignore_errors shouldn't raise, but guard anyway — we NEVER
            # want cleanup to crash a pipeline run.
            logger.exception("temp workspace cleanup failed: %s", workspace)
        else:
            logger.debug("temp workspace removed: %s", workspace)


def sweep_orphans(max_age_sec: int = _ORPHAN_MAX_AGE_SEC_DEFAULT) -> int:
    """
    Delete any `reelai-ingest-*` directories older than `max_age_sec` from the system temp
    directory. Returns the number of directories removed.

    Called once at module import and safe to call again (e.g. from a health-check endpoint).
    Never raises — errors are logged and swallowed, because a failing sweep must not block
    the worker from serving real requests.
    """
    removed = 0
    try:
        tempdir = Path(tempfile.gettempdir())
    except Exception:
        logger.exception("sweep_orphans: could not resolve tempdir")
        return 0

    now = time.time()
    try:
        entries = list(tempdir.iterdir())
    except FileNotFoundError:
        return 0
    except PermissionError:
        logger.warning("sweep_orphans: permission denied on tempdir %s", tempdir)
        return 0
    except Exception:
        logger.exception("sweep_orphans: iterdir failed on %s", tempdir)
        return 0

    for entry in entries:
        try:
            name = entry.name
            if not name.startswith(_WORKSPACE_PREFIX):
                continue
            if not entry.is_dir():
                continue
            stat = entry.stat()
            age = now - stat.st_mtime
            if age < max_age_sec:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except Exception:
            # Any failure on a single entry is logged but does not abort the sweep.
            logger.exception("sweep_orphans: failed to remove %s", entry)
            continue

    if removed:
        log_event(
            logger,
            logging.INFO,
            "orphan_sweep_complete",
            removed=removed,
            max_age_sec=max_age_sec,
            tempdir=str(tempdir),
        )
    return removed


# Run the sweep once at import time. This is the safety net for any workspace leaked by a
# prior crashed worker. Guard it under an env flag so tests can opt out by setting
# REELAI_INGEST_SKIP_IMPORT_SWEEP=1 before importing.
if not os.environ.get("REELAI_INGEST_SKIP_IMPORT_SWEEP"):
    try:
        sweep_orphans()
    except Exception:
        # sweep_orphans already catches everything, but belt-and-suspenders.
        logger.exception("import-time orphan sweep failed")


__all__ = ["TempWorkspace", "sweep_orphans"]
