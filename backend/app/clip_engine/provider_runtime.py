"""Shared provider budgets, retry parsing, and per-generation usage accounting."""
from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Literal, Mapping

from .errors import ProviderBudgetExceededError

ProviderOperation = Literal["search", "transcript", "segmentation"]
GenerationMode = Literal["fast", "slow"]

MAX_PROVIDER_RETRIES = 2
MAX_RETRY_AFTER_SEC = 30.0


def bounded_retry_after(
    headers: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
    maximum: float = MAX_RETRY_AFTER_SEC,
) -> float | None:
    """Parse Retry-After seconds or HTTP-date and clamp it to a safe ceiling."""
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    seconds: float | None = None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw).strip())
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            base = now or datetime.now(timezone.utc)
            seconds = (target - base).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if seconds is None or not math.isfinite(seconds):
        return None
    return max(0.0, min(float(maximum), seconds))


@dataclass(frozen=True)
class ProviderUsageRecord:
    provider: str
    operation: ProviderOperation
    attempt: int
    timestamp: str
    status_code: int | None = None
    billable_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_used: str = ""
    quality_degraded: bool = False
    error_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationBudget:
    """Atomic, job-wide provider ceilings plus acquisition-pass controls."""

    _LIMITS: dict[GenerationMode, dict[ProviderOperation, int]] = {
        "fast": {"search": 3, "transcript": 3, "segmentation": 3},
        "slow": {"search": 12, "transcript": 10, "segmentation": 10},
    }
    _PASS_LIMITS: dict[GenerationMode, tuple[int, int]] = {
        "fast": (1, 0),
        "slow": (3, 2),
    }

    def __init__(self, mode: GenerationMode) -> None:
        self.mode = mode
        self.limits = dict(self._LIMITS[mode])
        self.max_passes, self.max_no_growth_passes = self._PASS_LIMITS[mode]
        self._used: dict[ProviderOperation, int] = {
            "search": 0,
            "transcript": 0,
            "segmentation": 0,
        }
        self._passes = 0
        self._no_growth_passes = 0
        self._lock = threading.Lock()

    @classmethod
    def for_mode(cls, mode: str) -> "GenerationBudget":
        return cls("fast" if str(mode).strip().lower() == "fast" else "slow")

    def reserve(self, operation: ProviderOperation, count: int = 1) -> int:
        count = max(1, int(count))
        with self._lock:
            next_value = self._used[operation] + count
            if next_value > self.limits[operation]:
                raise ProviderBudgetExceededError(
                    f"{operation} budget exhausted ({self.limits[operation]} maximum).",
                    provider="generation",
                    operation=operation,
                )
            self._used[operation] = next_value
            return next_value

    def reserve_pass(self, *, no_growth: bool = False) -> int:
        with self._lock:
            if self._passes >= self.max_passes:
                raise ProviderBudgetExceededError(
                    f"acquisition-pass budget exhausted ({self.max_passes} maximum).",
                    provider="generation",
                    operation="search",
                )
            if no_growth and self._no_growth_passes >= self.max_no_growth_passes:
                raise ProviderBudgetExceededError(
                    "no-growth continuation budget exhausted.",
                    provider="generation",
                    operation="search",
                )
            self._passes += 1
            if no_growth:
                self._no_growth_passes += 1
            return self._passes

    def remaining(self, operation: ProviderOperation) -> int:
        with self._lock:
            return max(0, self.limits[operation] - self._used[operation])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "limits": dict(self.limits),
                "used": dict(self._used),
                "remaining": {
                    key: max(0, self.limits[key] - value)
                    for key, value in self._used.items()
                },
                "passes": self._passes,
                "max_passes": self.max_passes,
                "no_growth_passes": self._no_growth_passes,
                "max_no_growth_passes": self.max_no_growth_passes,
            }


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, Mapping) and name in usage:
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is None:
            continue
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
    return 0


class GenerationContext:
    """Thread-safe budget and usage ledger shared by all calls in one generation."""

    def __init__(
        self,
        mode: str,
        *,
        generation_id: str = "",
        usage_sink: Callable[[ProviderUsageRecord], None] | None = None,
        cache_store: Any = None,
    ) -> None:
        self.generation_id = generation_id
        self.budget = GenerationBudget.for_mode(mode)
        self.usage_sink = usage_sink
        self.cache_store = cache_store
        self._usage: list[ProviderUsageRecord] = []
        self._lock = threading.Lock()

    def reserve(self, operation: ProviderOperation) -> int:
        return self.budget.reserve(operation)

    def record(self, record: ProviderUsageRecord) -> None:
        with self._lock:
            self._usage.append(record)
        if self.usage_sink is not None:
            self.usage_sink(record)

    def record_http(
        self,
        *,
        provider: str,
        operation: ProviderOperation,
        attempt: int,
        status_code: int | None,
        headers: Mapping[str, Any] | None = None,
        error_code: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raw_billable = (headers or {}).get("x-billable-requests", 0)
        try:
            billable = max(0, int(raw_billable or 0))
        except (TypeError, ValueError):
            billable = 0
        record_metadata = dict(metadata or {})
        record_metadata["provider_call"] = True
        if (headers or {}).get("x-billable-requests") is not None:
            record_metadata["x_billable_requests"] = str(
                (headers or {}).get("x-billable-requests")
            )
        self.record(
            ProviderUsageRecord(
                provider=provider,
                operation=operation,
                attempt=max(1, int(attempt)),
                timestamp=datetime.now(timezone.utc).isoformat(),
                status_code=status_code,
                billable_requests=billable,
                error_code=error_code,
                metadata=record_metadata,
            )
        )

    def record_gemini(
        self,
        *,
        attempt: int,
        model_used: str,
        quality_degraded: bool,
        usage: Any = None,
        status_code: int | None = 200,
        error_code: str = "",
    ) -> None:
        input_tokens = _usage_value(
            usage, "prompt_token_count", "promptTokenCount", "input_tokens"
        )
        output_tokens = _usage_value(
            usage, "candidates_token_count", "candidatesTokenCount", "output_tokens"
        )
        total_tokens = _usage_value(usage, "total_token_count", "totalTokenCount", "total_tokens")
        self.record(
            ProviderUsageRecord(
                provider="gemini",
                operation="segmentation",
                attempt=max(1, int(attempt)),
                timestamp=datetime.now(timezone.utc).isoformat(),
                status_code=status_code,
                billable_requests=(
                    1 if status_code is not None and 200 <= status_code < 300 else 0
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or input_tokens + output_tokens,
                model_used=model_used,
                quality_degraded=quality_degraded,
                error_code=error_code,
                metadata={"provider_call": True},
            )
        )

    def usage(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.to_dict() for record in self._usage]
