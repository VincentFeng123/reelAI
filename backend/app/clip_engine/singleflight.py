"""Small keyed lock used to coalesce identical provider work in one process."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .cancellation import raise_if_cancelled


_locks_guard = threading.Lock()
_locks: dict[str, tuple[threading.Lock, int]] = {}


@contextmanager
def singleflight(
    key: str,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[None]:
    """Serialize identical cache-miss work while leaving unrelated keys parallel."""
    with _locks_guard:
        lock, users = _locks.get(key, (threading.Lock(), 0))
        _locks[key] = (lock, users + 1)

    acquired = False
    try:
        while not acquired:
            raise_if_cancelled(should_cancel)
            acquired = lock.acquire(timeout=0.25)
        yield
    finally:
        if acquired:
            lock.release()
        with _locks_guard:
            current_lock, current_users = _locks.get(key, (lock, 1))
            if current_lock is lock and current_users <= 1:
                _locks.pop(key, None)
            elif current_lock is lock:
                _locks[key] = (lock, current_users - 1)
