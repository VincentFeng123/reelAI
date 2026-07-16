"""Shared provider budgets, retry parsing, and per-generation usage accounting."""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Literal, Mapping

from .errors import ProviderBudgetExceededError

logger = logging.getLogger(__name__)

BudgetedProviderOperation = Literal["search", "transcript", "segmentation"]
ProviderOperation = Literal["search", "transcript", "segmentation", "expansion"]
GenerationMode = Literal["fast", "slow"]
GenerationCounter = Literal[
    "discovered_videos",
    "usable_transcripts",
    "transcript_failures",
    "transcript_timeouts",
    "clip_fetch_timeouts",
    "gemini_empty_results",
    "topic_rejections",
    "stored_clips",
    "deferred_clips",
    "persisted_clips",
    "verified_clips",
    "level_deferred_clips",
    "permanently_rejected_clips",
    "provider_failures",
    "provider_cursor_open",
    "segmentation_cache_hits",
    "expansion_cache_hits",
    "boundary_rejections",
    "boundary_unavailable",
    "boundary_repairs",
    "pro_fallbacks",
]

GENERATION_COUNTERS: tuple[GenerationCounter, ...] = (
    "discovered_videos",
    "usable_transcripts",
    "transcript_failures",
    "transcript_timeouts",
    "clip_fetch_timeouts",
    "gemini_empty_results",
    "topic_rejections",
    "stored_clips",
    "deferred_clips",
    "persisted_clips",
    "verified_clips",
    "level_deferred_clips",
    "permanently_rejected_clips",
    "provider_failures",
    "provider_cursor_open",
    "segmentation_cache_hits",
    "expansion_cache_hits",
    "boundary_rejections",
    "boundary_unavailable",
    "boundary_repairs",
    "pro_fallbacks",
)

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

    _LIMITS: dict[GenerationMode, dict[BudgetedProviderOperation, int]] = {
        # Reservations count provider attempts, including retries.
        "fast": {"search": 3, "transcript": 2, "segmentation": 2},
        "slow": {"search": 4, "transcript": 3, "segmentation": 3},
    }
    _PASS_LIMITS: dict[GenerationMode, tuple[int, int]] = {
        "fast": (1, 0),
        "slow": (1, 0),
    }
    _GEMINI_COST_LIMIT_USD: dict[GenerationMode, float] = {
        # Worst-case response reservations plus buffered input estimates cover
        # one Flash-Lite expansion and a bounded number of Pro clip selectors.
        # These are job ceilings, not expected spend; billed usage is recorded
        # from provider telemetry and is normally much lower.
        "fast": 0.45,
        "slow": 0.70,
    }
    _SELECTOR_CALL_LIMIT: dict[GenerationMode, int] = {
        "fast": 2,
        "slow": 3,
    }
    # Compatibility alias for older diagnostics/tests.  The active production
    # selector may now be Flash or Pro, but the bounded per-source call count is
    # shared so upgrading the model cannot multiply provider spend.
    _FLASH_SELECTOR_LIMIT = _SELECTOR_CALL_LIMIT
    _PRO_FALLBACK_CALL_LIMIT = 0

    def __init__(self, mode: GenerationMode) -> None:
        self.mode = mode
        self.limits = dict(self._LIMITS[mode])
        self.max_passes, self.max_no_growth_passes = self._PASS_LIMITS[mode]
        self._used: dict[BudgetedProviderOperation, int] = {
            "search": 0,
            "transcript": 0,
            "segmentation": 0,
        }
        self._passes = 0
        self._no_growth_passes = 0
        # Lifetime worst-case reservations remain a diagnostic. Admission uses
        # actual committed spend plus only the calls that are still in flight.
        self._gemini_reserved_cost_usd = 0.0
        self._gemini_committed_cost_usd = 0.0
        self._gemini_inflight: dict[int, float] = {}
        self._next_gemini_reservation_id = 1
        self._selector_calls = 0
        self._flash_selector_calls = 0
        self._pro_selector_calls = 0
        self._pro_fallback_calls = 0
        self._lock = threading.Lock()
        self._gemini_condition = threading.Condition(self._lock)

    @classmethod
    def for_mode(cls, mode: str) -> "GenerationBudget":
        return cls("fast" if str(mode).strip().lower() == "fast" else "slow")

    def reserve(self, operation: BudgetedProviderOperation, count: int = 1) -> int:
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

    def remaining(self, operation: BudgetedProviderOperation) -> int:
        with self._lock:
            return max(0, self.limits[operation] - self._used[operation])

    def reserve_gemini(
        self,
        *,
        model: str,
        operation: str,
        estimated_cost_usd: float,
        deadline_monotonic: float | None = None,
        cancelled: Callable[[], bool] | object | None = None,
    ) -> int:
        """Reserve a real Gemini dispatch, including nested selector fallbacks."""
        normalized_model = str(model or "").casefold()
        normalized_operation = str(operation or "").casefold()
        is_pro = "pro" in normalized_model or normalized_operation.startswith("pro_")
        is_pro_fallback = normalized_operation == "pro_fallback"
        is_pro_selector = is_pro and normalized_operation == "pro_authoritative"
        is_flash_selector = (
            not is_pro
            and normalized_operation
            in {
                "flash_single_candidate",
                "flash_boundary_selector",
                "boundary_selection",
            }
        )
        is_selector = is_flash_selector or is_pro_selector
        reservation = max(0.0, float(estimated_cost_usd))
        cost_limit = self._GEMINI_COST_LIMIT_USD[self.mode]

        def cancellation_requested() -> bool:
            if callable(cancelled):
                return bool(cancelled())
            if cancelled is None:
                return False
            is_set = getattr(cancelled, "is_set", None)
            return bool(is_set()) if callable(is_set) else bool(cancelled)

        while True:
            # Cancellation may touch external state, so never invoke it while
            # holding the budget lock needed by settlement and diagnostics.
            if cancellation_requested():
                raise ProviderBudgetExceededError(
                    "Gemini cost reservation cancelled.",
                    provider="gemini",
                    operation=operation,
                )
            if (
                deadline_monotonic is not None
                and float(deadline_monotonic) <= time.monotonic()
            ):
                raise ProviderBudgetExceededError(
                    "Gemini cost reservation deadline exceeded.",
                    provider="gemini",
                    operation=operation,
                )

            with self._gemini_condition:
                if is_pro and not (is_pro_selector or is_pro_fallback):
                    raise ProviderBudgetExceededError(
                        "This Gemini Pro operation is disabled for generation.",
                        provider="gemini",
                        operation=operation,
                    )
                if (
                    is_pro_fallback
                    and self._pro_fallback_calls >= self._PRO_FALLBACK_CALL_LIMIT
                ):
                    raise ProviderBudgetExceededError(
                        "Gemini Pro fallback is disabled.",
                        provider="gemini",
                        operation=operation,
                    )
                if (
                    is_selector
                    and self._selector_calls
                    >= self._SELECTOR_CALL_LIMIT[self.mode]
                ):
                    raise ProviderBudgetExceededError(
                        "Gemini transcript selector budget exhausted "
                        f"({self._SELECTOR_CALL_LIMIT[self.mode]} maximum).",
                        provider="gemini",
                        operation=operation,
                    )
                inflight_cost = sum(self._gemini_inflight.values())
                exposure = self._gemini_committed_cost_usd + inflight_cost
                if exposure + reservation <= cost_limit + 1e-9:
                    reservation_id = self._next_gemini_reservation_id
                    self._next_gemini_reservation_id += 1
                    self._gemini_inflight[reservation_id] = reservation
                    self._gemini_reserved_cost_usd += reservation
                    if is_pro_fallback:
                        self._pro_fallback_calls += 1
                    if is_selector:
                        self._selector_calls += 1
                    if is_flash_selector:
                        self._flash_selector_calls += 1
                    if is_pro_selector:
                        self._pro_selector_calls += 1
                    return reservation_id

                # Settling in-flight work cannot reduce already committed
                # spend, so this request can never fit within the job ceiling.
                if self._gemini_committed_cost_usd + reservation > cost_limit + 1e-9:
                    raise ProviderBudgetExceededError(
                        f"Gemini job cost budget exhausted (${cost_limit:.2f} maximum).",
                        provider="gemini",
                        operation=operation,
                        detail=(
                            f"committed=${self._gemini_committed_cost_usd:.6f}, "
                            f"inflight=${inflight_cost:.6f}, requested=${reservation:.6f}"
                        ),
                    )
                if not self._gemini_inflight or deadline_monotonic is None:
                    raise ProviderBudgetExceededError(
                        f"Gemini job cost budget exhausted (${cost_limit:.2f} maximum).",
                        provider="gemini",
                        operation=operation,
                        detail=(
                            f"committed=${self._gemini_committed_cost_usd:.6f}, "
                            f"inflight=${inflight_cost:.6f}, requested=${reservation:.6f}"
                        ),
                    )
                remaining = float(deadline_monotonic) - time.monotonic()
                if remaining <= 0:
                    continue
                self._gemini_condition.wait(timeout=min(0.05, remaining))

    def reconcile_gemini(
        self,
        reservation_id: int | None,
        *,
        actual_cost_usd: float | None,
    ) -> bool:
        """Replace one in-flight worst case with actual cost; unknown billing keeps it."""
        with self._gemini_condition:
            if reservation_id is None:
                if actual_cost_usd is not None:
                    self._gemini_committed_cost_usd += max(
                        0.0, float(actual_cost_usd)
                    )
                return False
            reserved = self._gemini_inflight.pop(int(reservation_id), None)
            if reserved is None:
                return False
            committed = (
                reserved
                if actual_cost_usd is None
                else max(0.0, float(actual_cost_usd))
            )
            self._gemini_committed_cost_usd += committed
            if committed > reserved + 1e-9:
                logger.warning(
                    "Gemini actual cost exceeded reservation: actual=$%.6f reserved=$%.6f",
                    committed,
                    reserved,
                )
            self._gemini_condition.notify_all()
            return True

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
                "gemini": {
                    "cost_limit_usd": self._GEMINI_COST_LIMIT_USD[self.mode],
                    "lifetime_reserved_worst_case_cost_usd": round(
                        self._gemini_reserved_cost_usd, 8
                    ),
                    # Compatibility alias for cumulative historical maxima.
                    # It is a reservation diagnostic, never billed spend; use
                    # cost_exposure_usd for the current admission exposure.
                    "reserved_cost_usd": round(self._gemini_reserved_cost_usd, 8),
                    "settled_cost_exposure_usd": round(
                        self._gemini_committed_cost_usd, 8
                    ),
                    "committed_cost_usd": round(self._gemini_committed_cost_usd, 8),
                    "inflight_reserved_cost_usd": round(
                        sum(self._gemini_inflight.values()), 8
                    ),
                    "cost_exposure_usd": round(
                        self._gemini_committed_cost_usd
                        + sum(self._gemini_inflight.values()),
                        8,
                    ),
                    "selector_calls": self._selector_calls,
                    "selector_limit": self._SELECTOR_CALL_LIMIT[self.mode],
                    "flash_selector_calls": self._flash_selector_calls,
                    "flash_selector_limit": self._SELECTOR_CALL_LIMIT[self.mode],
                    "pro_selector_calls": self._pro_selector_calls,
                    "pro_selector_limit": self._SELECTOR_CALL_LIMIT[self.mode],
                    # Keep the older names as compatibility aliases. These
                    # counters have always represented the fallback allowance,
                    # not ordinary authoritative Pro selectors.
                    "pro_calls": self._pro_fallback_calls,
                    "pro_call_limit": self._PRO_FALLBACK_CALL_LIMIT,
                    "pro_fallback_calls": self._pro_fallback_calls,
                    "pro_fallback_call_limit": self._PRO_FALLBACK_CALL_LIMIT,
                },
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


def _usage_field(usage: Any, name: str) -> Any:
    if isinstance(usage, Mapping):
        return usage.get(name)
    return getattr(usage, name, None)


def _gemini_token_rates(
    model: str,
    *,
    input_tokens: int = 0,
) -> tuple[float, float, float]:
    """Return current per-million uncached/cached-input/output rates."""
    normalized = str(model or "").casefold()
    if "flash-lite" in normalized:
        return 0.25, 0.025, 1.50
    if "gemini-3-flash" in normalized and "gemini-3.5-flash" not in normalized:
        return 0.50, 0.05, 3.00
    if "pro" in normalized:
        if max(0, int(input_tokens)) > 200_000:
            return 4.00, 0.40, 18.00
        return 2.00, 0.20, 12.00
    return 1.50, 0.15, 9.00


def _gemini_physical_attempts(record: Mapping[str, Any]) -> int:
    """Count a logical call's transport attempts without inventing token usage."""
    retries = (record.get("metadata") or {}).get("retries")
    if retries is None:
        return 1
    try:
        return max(1, int(retries) + 1)
    except (TypeError, ValueError, OverflowError):
        return 1


class GenerationContext:
    """Thread-safe budget and usage ledger shared by all calls in one generation."""

    def __init__(
        self,
        mode: str,
        *,
        generation_id: str = "",
        usage_sink: Callable[[ProviderUsageRecord], None] | None = None,
        cache_store: Any = None,
        require_acoustic_boundaries: bool = False,
    ) -> None:
        self.generation_id = generation_id
        self.budget = GenerationBudget.for_mode(mode)
        self.usage_sink = usage_sink
        self.cache_store = cache_store
        self.require_acoustic_boundaries = bool(require_acoustic_boundaries)
        self._usage: list[ProviderUsageRecord] = []
        self._counters: dict[GenerationCounter, int] = {
            name: 0 for name in GENERATION_COUNTERS
        }
        self._fallback_reasons: list[str] = []
        self._rejection_reason_counts: dict[str, int] = {}
        self._counted_pro_fallbacks: set[tuple[str, bool]] = set()
        self._lock = threading.Lock()
        self._pro_fallback_condition = threading.Condition(self._lock)
        self._pro_fallback_cohort_size = 1
        self._pro_fallback_gate_configured = False
        self._pro_fallback_cohort_results: dict[int, tuple[int, bool]] = {}
        self._pro_fallback_claimed = False

    def reserve(self, operation: BudgetedProviderOperation) -> int:
        return self.budget.reserve(operation)

    def reserve_gemini_call(
        self,
        *,
        operation: str,
        model: str,
        max_output_tokens: int,
        prompt_text: str = "",
        estimated_input_tokens: int | None = None,
        deadline_monotonic: float | None = None,
        cancelled: Callable[[], bool] | object | None = None,
    ) -> dict[str, int | float]:
        """Reserve worst-case billed tokens before a Gemini request is dispatched."""
        prompt_tokens = (
            max(0, int(estimated_input_tokens))
            if estimated_input_tokens is not None
            else max(1, math.ceil(len(str(prompt_text or "")) / 4))
        )
        output_tokens = max(1, int(max_output_tokens))
        input_rate, _cached_input_rate, output_rate = _gemini_token_rates(
            model,
            input_tokens=prompt_tokens,
        )
        estimated_cost = (
            prompt_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000.0
        reservation_id = self.budget.reserve_gemini(
            model=model,
            operation=operation,
            estimated_cost_usd=estimated_cost,
            deadline_monotonic=deadline_monotonic,
            cancelled=cancelled,
        )
        return {
            "gemini_reservation_id": reservation_id,
            "reserved_input_tokens": prompt_tokens,
            "reserved_output_tokens": output_tokens,
            "reserved_cost_usd": estimated_cost,
        }

    def reconcile_gemini_call(
        self,
        *,
        model_used: str,
        usage: Any = None,
        dispatched: bool | None = None,
    ) -> bool:
        """Settle one reservation as soon as provider telemetry is available."""
        raw_id = _usage_field(usage, "gemini_reservation_id")
        try:
            reservation_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError, OverflowError):
            reservation_id = None
        input_tokens = _usage_value(
            usage,
            "prompt_tokens",
            "prompt_token_count",
            "promptTokenCount",
            "input_tokens",
        )
        candidate_tokens = _usage_value(
            usage,
            "candidate_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
        )
        thought_tokens = _usage_value(
            usage,
            "thought_tokens",
            "thoughts_token_count",
            "thoughtsTokenCount",
        )
        output_tokens = (
            candidate_tokens + thought_tokens
            if candidate_tokens or thought_tokens
            else _usage_value(usage, "output_tokens")
        )
        candidate_usage_present = any(
            _usage_field(usage, name) is not None
            for name in (
                "candidate_tokens",
                "candidates_token_count",
                "candidatesTokenCount",
            )
        )
        thought_usage_present = any(
            _usage_field(usage, name) is not None
            for name in (
                "thought_tokens",
                "thoughts_token_count",
                "thoughtsTokenCount",
            )
        )
        aggregate_output_present = _usage_field(usage, "output_tokens") is not None
        cached_tokens = min(
            input_tokens,
            _usage_value(
                usage,
                "cached_tokens",
                "cached_content_token_count",
                "cachedContentTokenCount",
            ),
        )
        # A total without the input/output split cannot be priced correctly;
        # retain the full reservation rather than reopening the budget at $0.
        # Every Gemini call has a non-empty prompt. Price actual usage only
        # when both sides of the split are present. Present zero-valued output
        # counters are complete; absent input/output counters are partial
        # telemetry and must retain the full fail-closed reservation.
        usage_known = input_tokens > 0 and (
            aggregate_output_present
            or (candidate_usage_present and thought_usage_present)
        )
        dispatched_value = (
            bool(_usage_field(usage, "dispatched"))
            if dispatched is None
            else bool(dispatched)
        )
        actual_cost: float | None
        if usage_known:
            input_rate, cached_input_rate, output_rate = _gemini_token_rates(
                model_used,
                input_tokens=input_tokens,
            )
            actual_cost = (
                (input_tokens - cached_tokens) * input_rate
                + cached_tokens * cached_input_rate
                + output_tokens * output_rate
            ) / 1_000_000.0
        elif dispatched_value:
            # A dispatched call with missing token telemetry may still be billed.
            actual_cost = None
        else:
            actual_cost = 0.0
        return self.budget.reconcile_gemini(
            reservation_id,
            actual_cost_usd=actual_cost,
        )

    def configure_pro_fallback_gate(self, expected_initial_results: int) -> None:
        """Set the first-wave size before its concurrent selectors are started."""
        expected = max(1, min(2, int(expected_initial_results)))
        with self._pro_fallback_condition:
            if (
                not self._pro_fallback_gate_configured
                and not self._pro_fallback_cohort_results
            ):
                self._pro_fallback_cohort_size = expected
                self._pro_fallback_gate_configured = True

    def record_pro_fallback_cohort_result(
        self,
        *,
        candidate_rank: int,
        accepted_count: int,
        fallback_eligible: bool,
    ) -> None:
        """Resolve an initial video that never reaches the Flash fallback gate."""
        rank = max(0, int(candidate_rank))
        accepted = max(0, int(accepted_count))
        with self._pro_fallback_condition:
            if rank < self._pro_fallback_cohort_size:
                self._pro_fallback_cohort_results[rank] = (
                    accepted,
                    bool(fallback_eligible and accepted == 0),
                )
                self._pro_fallback_condition.notify_all()

    def allow_pro_fallback(
        self,
        *,
        accepted_count: int,
        video_id: str = "",
        candidate_rank: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> bool:
        """Claim Pro for the best failed candidate after joint initial yield is known.

        Failed first-wave selectors wait for their peer's result, so completion
        order cannot decide which video spends the one fallback. Successful
        selectors report and return immediately, preserving the fast path.
        """
        del video_id  # Reserved for usage diagnostics; selection is rank-based.
        accepted = max(0, int(accepted_count))
        with self._pro_fallback_condition:
            rank = (
                max(0, int(candidate_rank))
                if candidate_rank is not None
                else len(self._pro_fallback_cohort_results)
            )
            if rank < self._pro_fallback_cohort_size:
                self._pro_fallback_cohort_results.setdefault(
                    rank,
                    (accepted, accepted == 0),
                )
                self._pro_fallback_condition.notify_all()

            if accepted > 0:
                return False

            if rank < self._pro_fallback_cohort_size:
                wait_deadline = (
                    float(deadline_monotonic)
                    if deadline_monotonic is not None
                    else time.monotonic() + 30.0
                )
                while (
                    len(self._pro_fallback_cohort_results)
                    < self._pro_fallback_cohort_size
                ):
                    remaining = wait_deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._pro_fallback_condition.wait(timeout=remaining)

            initial_accepted = sum(
                result[0] for result in self._pro_fallback_cohort_results.values()
            )
            if initial_accepted >= 3 or self._pro_fallback_claimed:
                return False
            if rank < self._pro_fallback_cohort_size:
                eligible_initial = [
                    initial_rank
                    for initial_rank, (count, eligible) in
                    self._pro_fallback_cohort_results.items()
                    if count == 0 and eligible
                ]
                if not eligible_initial or rank != min(eligible_initial):
                    return False
            self._pro_fallback_claimed = True
            return True

    def claim_aggregate_pro_fallback(self, *, validated_count: int) -> bool:
        """Claim leftover fallback after both initial videos pass topic validation."""
        with self._pro_fallback_condition:
            if max(0, int(validated_count)) >= 3 or self._pro_fallback_claimed:
                return False
            self._pro_fallback_claimed = True
            return True

    def record(self, record: ProviderUsageRecord) -> None:
        with self._lock:
            self._usage.append(record)
        if self.usage_sink is not None:
            try:
                self.usage_sink(record)
            except Exception as exc:
                logger.warning("Provider usage persistence failed: %s", exc)

    def increment_counter(self, name: GenerationCounter, count: int = 1) -> int:
        """Atomically increment a generation-stage diagnostic counter."""
        if name not in self._counters:
            raise ValueError(f"unknown generation counter: {name}")
        amount = int(count)
        if amount < 0:
            raise ValueError("generation counters cannot be decremented")
        with self._lock:
            self._counters[name] += amount
            return self._counters[name]

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def record_segment_event(self, event: Mapping[str, Any]) -> None:
        """Collect non-billable clipping diagnostics for the generation summary."""
        event_name = str(event.get("event") or "")
        if event_name == "pro_fallback":
            fallback_key = (
                str(event.get("video_id") or "__job__"),
                bool(event.get("shadow", False)),
            )
            raw_reasons = event.get("reasons")
            reasons = (
                [str(reason) for reason in raw_reasons if str(reason).strip()]
                if isinstance(raw_reasons, list)
                else []
            )
            reason = str(event.get("reason") or "").strip()
            if reason:
                reasons.append(reason)
            if not reasons:
                reasons = ["unknown"]
            with self._lock:
                first_event = fallback_key not in self._counted_pro_fallbacks
                self._counted_pro_fallbacks.add(fallback_key)
                for fallback_reason in reasons:
                    if fallback_reason not in self._fallback_reasons:
                        self._fallback_reasons.append(fallback_reason)
            if first_event:
                self.increment_counter("pro_fallbacks")
        elif event_name == "boundary_repair":
            self.increment_counter("boundary_repairs")
        elif event_name == "flash_classified":
            proposed = max(0, int(event.get("proposed_count") or 0))
            accepted = max(0, int(event.get("accepted_count") or 0))
            if proposed > accepted:
                self.increment_counter("boundary_rejections", proposed - accepted)
        if event_name in {"segment_completed", "segment_error"}:
            raw_reasons = event.get("rejection_reasons")
            reasons = (
                [str(reason).strip() for reason in raw_reasons if str(reason).strip()]
                if isinstance(raw_reasons, list)
                else []
            )
            with self._lock:
                for reason in reasons:
                    prefix, separator, guard = reason.partition(":")
                    normalized = (
                        guard
                        if separator
                        and (prefix.startswith("proposal_") or prefix.startswith("candidate_"))
                        else reason
                    )
                    self._rejection_reason_counts[normalized] = (
                        self._rejection_reason_counts.get(normalized, 0) + 1
                    )

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

    def record_cache_hit(
        self,
        *,
        provider: str,
        operation: ProviderOperation,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record reusable provider output without counting a provider request."""
        record_metadata = dict(metadata or {})
        record_metadata.update({"provider_call": False, "cache_hit": True})
        self.record(
            ProviderUsageRecord(
                provider=provider,
                operation=operation,
                attempt=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=record_metadata,
            )
        )

    def record_gemini(
        self,
        *,
        operation: ProviderOperation = "segmentation",
        attempt: int,
        model_used: str,
        quality_degraded: bool,
        usage: Any = None,
        status_code: int | None = 200,
        error_code: str = "",
        stage: str = "",
    ) -> None:
        input_tokens = _usage_value(
            usage,
            "prompt_tokens",
            "prompt_token_count",
            "promptTokenCount",
            "input_tokens",
        )
        candidate_tokens = _usage_value(
            usage,
            "candidate_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
        )
        thought_tokens = _usage_value(
            usage,
            "thought_tokens",
            "thoughts_token_count",
            "thoughtsTokenCount",
        )
        cached_tokens = _usage_value(
            usage,
            "cached_tokens",
            "cached_content_token_count",
            "cachedContentTokenCount",
        )
        output_tokens = (
            candidate_tokens + thought_tokens
            if candidate_tokens or thought_tokens
            else _usage_value(usage, "output_tokens")
        )
        total_tokens = _usage_value(usage, "total_token_count", "totalTokenCount", "total_tokens")
        record_metadata: dict[str, Any] = {
            "provider_call": True,
            "candidate_tokens": candidate_tokens,
            "thought_tokens": thought_tokens,
            "cached_tokens": cached_tokens,
        }
        if stage:
            record_metadata["stage"] = str(stage)
        for field_name in (
            "video_id",
            "video_grounded",
            "latency_ms",
            "retries",
            "finish_reason",
            "prompt_version",
            "thinking_level",
            "gemini_reservation_id",
            "reserved_input_tokens",
            "reserved_output_tokens",
            "reserved_cost_usd",
            "dispatched",
            "error_type",
            "provider_error_type",
            "provider_status_code",
            "retryable",
            "token_preflight_failed",
            "error_history",
            "failover_from_model",
            "failover_model",
            "failover_reason",
        ):
            value = _usage_field(usage, field_name)
            if value is not None:
                record_metadata[field_name] = value
        record = ProviderUsageRecord(
            provider="gemini",
            operation=operation,
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
            metadata=record_metadata,
        )
        self.reconcile_gemini_call(
            model_used=model_used,
            # Reconcile from the original provider payload so absent split
            # fields remain distinguishable from explicit zero counters. The
            # normalized ledger intentionally stores zeros for aggregation,
            # but those synthesized zeros are not billing evidence.
            usage=usage,
            dispatched=(
                bool(record_metadata.get("dispatched"))
                if "dispatched" in record_metadata
                else status_code is not None
            ),
        )
        self.record(record)

    def usage(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.to_dict() for record in self._usage]

    @staticmethod
    def _gemini_cost(record: Mapping[str, Any]) -> float:
        input_tokens = max(0, int(record.get("input_tokens") or 0))
        input_rate, cached_input_rate, output_rate = _gemini_token_rates(
            str(record.get("model_used") or ""),
            input_tokens=input_tokens,
        )
        cached_tokens = min(
            input_tokens,
            max(0, int((record.get("metadata") or {}).get("cached_tokens") or 0)),
        )
        uncached_tokens = input_tokens - cached_tokens
        return (
            uncached_tokens * input_rate
            + cached_tokens * cached_input_rate
            + int(record.get("output_tokens") or 0) * output_rate
        ) / 1_000_000.0

    def usage_payload(self) -> dict[str, Any]:
        """Stable, aggregated job usage plus the existing raw provider ledger."""
        records = self.usage()
        counters = self.counters()
        gemini_calls = [
            row
            for row in records
            if row.get("provider") == "gemini"
            and bool((row.get("metadata") or {}).get("provider_call"))
        ]
        by_stage: dict[str, dict[str, int | float]] = {}
        for row in gemini_calls:
            metadata = row.get("metadata") or {}
            stage = str(metadata.get("stage") or row.get("operation") or "unknown")
            bucket = by_stage.setdefault(
                stage,
                {
                    "calls": 0,
                    "attempts": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thought_tokens": 0,
                    "cached_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "reserved_cost_usd": 0.0,
                    "billing_unknown_calls": 0,
                },
            )
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["attempts"] = int(bucket["attempts"]) + (
                _gemini_physical_attempts(row)
            )
            bucket["input_tokens"] = int(bucket["input_tokens"]) + int(
                row.get("input_tokens") or 0
            )
            bucket["output_tokens"] = int(bucket["output_tokens"]) + int(
                row.get("output_tokens") or 0
            )
            bucket["thought_tokens"] = int(bucket["thought_tokens"]) + int(
                metadata.get("thought_tokens") or 0
            )
            bucket["cached_tokens"] = int(bucket["cached_tokens"]) + int(
                metadata.get("cached_tokens") or 0
            )
            bucket["estimated_cost_usd"] = (
                float(bucket["estimated_cost_usd"]) + self._gemini_cost(row)
            )
            bucket["reserved_cost_usd"] = (
                float(bucket["reserved_cost_usd"])
                + float(metadata.get("reserved_cost_usd") or 0.0)
            )
            if bool(metadata.get("dispatched")) and not int(row.get("total_tokens") or 0):
                bucket["billing_unknown_calls"] = int(
                    bucket["billing_unknown_calls"]
                ) + 1
        for bucket in by_stage.values():
            bucket["estimated_cost_usd"] = round(
                float(bucket["estimated_cost_usd"]), 8
            )
            bucket["reserved_cost_usd"] = round(
                float(bucket["reserved_cost_usd"]), 8
            )

        estimated_cost = sum(self._gemini_cost(row) for row in gemini_calls)
        accepted = int(counters.get("persisted_clips") or 0)
        with self._lock:
            fallback_reasons = list(self._fallback_reasons)
            rejection_reason_counts = dict(sorted(self._rejection_reason_counts.items()))
        cache_hits = sum(
            1 for row in records if bool((row.get("metadata") or {}).get("cache_hit"))
        )
        budget_snapshot = self.budget.snapshot()["gemini"]
        summary = {
            "gemini_calls": len(gemini_calls),
            "gemini_attempts": sum(
                _gemini_physical_attempts(row) for row in gemini_calls
            ),
            "flash_calls": sum(
                1
                for row in gemini_calls
                if "pro" not in str(row.get("model_used") or "").casefold()
            ),
            "pro_calls": sum(
                1
                for row in gemini_calls
                if "pro" in str(row.get("model_used") or "").casefold()
            ),
            "input_tokens": sum(
                int(row.get("input_tokens") or 0) for row in gemini_calls
            ),
            "output_tokens": sum(
                int(row.get("output_tokens") or 0) for row in gemini_calls
            ),
            "thought_tokens": sum(
                int((row.get("metadata") or {}).get("thought_tokens") or 0)
                for row in gemini_calls
            ),
            "cached_tokens": sum(
                int((row.get("metadata") or {}).get("cached_tokens") or 0)
                for row in gemini_calls
            ),
            "estimated_cost_usd": round(estimated_cost, 8),
            "current_cost_exposure_usd": budget_snapshot[
                "cost_exposure_usd"
            ],
            "cost_limit_usd": budget_snapshot["cost_limit_usd"],
            # Compatibility alias for lifetime reservation history. It can
            # exceed the job ceiling after sequential calls and is not spend;
            # current_cost_exposure_usd is the active bounded value.
            "reserved_worst_case_cost_usd": budget_snapshot[
                "lifetime_reserved_worst_case_cost_usd"
            ],
            "lifetime_reserved_worst_case_cost_usd": budget_snapshot[
                "lifetime_reserved_worst_case_cost_usd"
            ],
            "billing_unknown_calls": sum(
                1
                for row in gemini_calls
                if bool((row.get("metadata") or {}).get("dispatched"))
                and not int(row.get("total_tokens") or 0)
            ),
            "accepted_clips": accepted,
            "cost_per_accepted_clip_usd": (
                round(estimated_cost / accepted, 8) if accepted else None
            ),
            "cache_hits": cache_hits,
            "fallback_reasons": fallback_reasons,
            "rejection_reason_counts": rejection_reason_counts,
            "rejected_boundaries": int(counters.get("boundary_rejections") or 0),
        }
        return {
            "budget": self.budget.snapshot(),
            "summary": summary,
            "by_stage": by_stage,
            "provider_calls": records,
            "counters": counters,
        }
