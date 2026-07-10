"""Small async-to-sync bridge for actively cancellable provider requests.

The public clipping pipeline remains synchronous because generation currently
runs in worker threads. Provider I/O is async internally so a caller probe can
cancel the in-flight socket instead of merely ignoring its eventual response.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import CancellationError

T = TypeVar("T")
CancellationProbe = Callable[[], bool] | None
POLL_INTERVAL_SEC = 0.05


def is_cancelled(probe: CancellationProbe) -> bool:
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def raise_if_cancelled(probe: CancellationProbe) -> None:
    if is_cancelled(probe):
        raise CancellationError("Generation cancelled.")


async def _await_with_probe(awaitable: Awaitable[T], probe: CancellationProbe) -> T:
    raise_if_cancelled(probe)
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=POLL_INTERVAL_SEC)
            if task in done:
                return await task
            if is_cancelled(probe):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise CancellationError("Generation cancelled.")
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


def run_cancellable(factory: Callable[[], Awaitable[T]], probe: CancellationProbe = None) -> T:
    """Run an async request from synchronous pipeline code.

    Normal generation workers have no running event loop. The helper-thread
    branch keeps the function safe for unit tests or future async call sites.
    """
    raise_if_cancelled(probe)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_with_probe(factory(), probe))

    result: list[T] = []
    error: list[BaseException] = []
    finished = threading.Event()

    def runner() -> None:
        try:
            result.append(asyncio.run(_await_with_probe(factory(), probe)))
        except BaseException as exc:  # re-raised in the calling thread
            error.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    while not finished.wait(POLL_INTERVAL_SEC):
        # The async helper observes the same probe and cancels its task. Keeping
        # this wait bounded also makes cancellation visible to the sync caller.
        continue
    if error:
        raise error[0]
    return result[0]


async def sleep_with_probe(seconds: float, probe: CancellationProbe = None) -> None:
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        raise_if_cancelled(probe)
        step = min(POLL_INTERVAL_SEC, remaining)
        await asyncio.sleep(step)
        remaining -= step
    raise_if_cancelled(probe)


def wait_with_probe(seconds: float, probe: CancellationProbe = None) -> None:
    run_cancellable(lambda: sleep_with_probe(seconds, probe), probe)
