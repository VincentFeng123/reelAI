"""
Client-level circuit breaker (Phase B.3).

Used to gate yt-dlp `fallback_clients` (and other rotatable identities like
proxy nodes, Innertube clients, data-API keys) based on recent failure history.
A client in the `open` state is skipped until its cooldown elapses and the
breaker moves to `half_open`, allowing exactly one probe call; success closes
the breaker, failure re-opens it with a longer cooldown.

Design:
  - Thread-safe (threading.Lock).
  - In-process only — not shared across Railway replicas. That's fine for the
    current single-replica deploy; if the app ever scales horizontally, swap
    the state backend to Redis.
  - No external timer — state transitions are lazy, computed on every
    `allow()` call from the current clock vs. `opened_at`.
  - Two knobs per client: `failure_threshold` (open after N consecutive or
    in-window failures) and `cooldown_sec` (how long to stay open). Defaults
    tuned for yt-dlp bot-detection patterns (quick open, slow recovery).

Usage:
    breaker = CircuitBreaker(failure_threshold=5, cooldown_sec=300.0)
    for client in available_clients:
        if not breaker.allow(client):
            continue
        try:
            result = do_work(client)
            breaker.record_success(client)
            return result
        except TransientError:
            breaker.record_failure(client)
            continue
    raise AllClientsDown()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _ClientState:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    opened_at_mono: float = 0.0
    last_failure_mono: float = 0.0
    total_opens: int = 0
    # Dynamic cooldown — each re-open after half-open failure uses 1.5x the prior cooldown,
    # capped at `max_cooldown_sec`.
    current_cooldown_sec: float = 0.0
    # Probes issued since the breaker went half-open — allows the `allow()`
    # check to cap concurrent probes without waiting for record_success.
    probes_issued: int = 0


class CircuitBreaker:
    """Multi-client circuit breaker keyed on a string client identifier."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        failure_window_sec: float = 60.0,
        initial_cooldown_sec: float = 300.0,
        max_cooldown_sec: float = 3600.0,
        half_open_probe_count: int = 1,
        success_threshold_to_close: int = 2,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        # Minimum 0.05s so unit tests can drive sub-second cooldowns; production
        # callers pass values ≥ 60s which are unaffected by this floor.
        self.failure_window_sec = max(0.05, failure_window_sec)
        self.initial_cooldown_sec = max(0.05, initial_cooldown_sec)
        self.max_cooldown_sec = max(self.initial_cooldown_sec, max_cooldown_sec)
        self.half_open_probe_count = max(1, half_open_probe_count)
        self.success_threshold_to_close = max(1, success_threshold_to_close)
        self._states: dict[str, _ClientState] = {}
        self._lock = threading.Lock()

    # ---- state inspection ------------------------------------------------- #

    def state(self, client: str) -> BreakerState:
        with self._lock:
            cs = self._states.get(client)
            if cs is None:
                return BreakerState.CLOSED
            self._maybe_transition_to_half_open_locked(cs)
            return cs.state

    def snapshot(self) -> dict[str, dict]:
        """Dump all tracked clients' state — for /admin/health observability."""
        with self._lock:
            out: dict[str, dict] = {}
            for client, cs in self._states.items():
                out[client] = {
                    "state": cs.state.value,
                    "consecutive_failures": cs.consecutive_failures,
                    "opened_at_mono": cs.opened_at_mono,
                    "total_opens": cs.total_opens,
                    "current_cooldown_sec": cs.current_cooldown_sec,
                }
            return out

    # ---- decision --------------------------------------------------------- #

    def allow(self, client: str) -> bool:
        """
        Return True if work should be attempted on `client` right now.
        A client in `half_open` state returns True at most `half_open_probe_count`
        times before other callers see False until the probe resolves.
        """
        with self._lock:
            cs = self._states.setdefault(client, _ClientState())
            self._maybe_transition_to_half_open_locked(cs)

            if cs.state == BreakerState.CLOSED:
                return True
            if cs.state == BreakerState.OPEN:
                return False
            # HALF_OPEN: allow at most `half_open_probe_count` callers through.
            # We count probes at allow-time (not record_*-time) so concurrent
            # callers are gated even before the in-flight probe resolves.
            if cs.probes_issued < self.half_open_probe_count:
                cs.probes_issued += 1
                return True
            return False

    # ---- feedback --------------------------------------------------------- #

    def record_success(self, client: str) -> None:
        with self._lock:
            cs = self._states.setdefault(client, _ClientState())
            if cs.state == BreakerState.HALF_OPEN:
                cs.consecutive_successes += 1
                cs.consecutive_failures = 0
                if cs.consecutive_successes >= self.success_threshold_to_close:
                    logger.info(
                        "circuit_breaker closing for client=%s after %d probe successes",
                        client, cs.consecutive_successes,
                    )
                    cs.state = BreakerState.CLOSED
                    cs.consecutive_successes = 0
                    cs.consecutive_failures = 0
                    cs.current_cooldown_sec = 0.0
                return
            if cs.state == BreakerState.CLOSED:
                cs.consecutive_failures = 0

    def record_failure(self, client: str, *, error_class: str | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            cs = self._states.setdefault(client, _ClientState())
            if cs.state == BreakerState.HALF_OPEN:
                # Half-open probe failed → re-open with escalated cooldown.
                cs.state = BreakerState.OPEN
                cs.opened_at_mono = now
                cs.total_opens += 1
                cs.consecutive_failures = self.failure_threshold  # stay tripped
                cs.consecutive_successes = 0
                cs.current_cooldown_sec = min(
                    self.max_cooldown_sec,
                    max(self.initial_cooldown_sec, cs.current_cooldown_sec) * 1.5,
                )
                logger.warning(
                    "circuit_breaker re-opening for client=%s after half-open failure err=%s cooldown=%.0fs",
                    client, error_class, cs.current_cooldown_sec,
                )
                return
            # Count failures only inside the rolling window.
            if (now - cs.last_failure_mono) > self.failure_window_sec:
                cs.consecutive_failures = 0
            cs.consecutive_failures += 1
            cs.last_failure_mono = now
            if cs.state == BreakerState.CLOSED and cs.consecutive_failures >= self.failure_threshold:
                cs.state = BreakerState.OPEN
                cs.opened_at_mono = now
                cs.total_opens += 1
                cs.current_cooldown_sec = self.initial_cooldown_sec
                logger.warning(
                    "circuit_breaker opening for client=%s after %d failures in %.0fs err=%s",
                    client, cs.consecutive_failures, self.failure_window_sec, error_class,
                )

    # ---- helpers ---------------------------------------------------------- #

    def _maybe_transition_to_half_open_locked(self, cs: _ClientState) -> None:
        if cs.state != BreakerState.OPEN:
            return
        cooldown = cs.current_cooldown_sec or self.initial_cooldown_sec
        if (time.monotonic() - cs.opened_at_mono) >= cooldown:
            cs.state = BreakerState.HALF_OPEN
            cs.consecutive_successes = 0
            cs.consecutive_failures = 0
            cs.probes_issued = 0
            logger.info(
                "circuit_breaker transitioning to HALF_OPEN after %.0fs cooldown",
                cooldown,
            )


# --------------------------------------------------------------------------- #
# Global singletons used by the yt-dlp adapter (Phase B.3 integration point)
# --------------------------------------------------------------------------- #

# The three yt-dlp client identities we currently round-robin through.
ytdlp_client_breaker = CircuitBreaker(
    failure_threshold=5,
    failure_window_sec=60.0,
    initial_cooldown_sec=300.0,
    max_cooldown_sec=1800.0,
)


__all__ = [
    "BreakerState",
    "CircuitBreaker",
    "ytdlp_client_breaker",
]
