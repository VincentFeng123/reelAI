"""Shared subscription entitlements and transactional daily search quotas."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ..db import execute_modify, fetch_all, fetch_one, insert, upsert


PLAN_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "code": "free",
        "name": "Free",
        "monthly_price_cents": 0,
        "daily_limit": 5,
    },
    {
        "code": "plus",
        "name": "Plus",
        "monthly_price_cents": 499,
        "daily_limit": 15,
    },
    {
        "code": "pro",
        "name": "Pro",
        "monthly_price_cents": 1999,
        "daily_limit": 50,
    },
)
PLAN_BY_CODE = {str(plan["code"]): plan for plan in PLAN_CATALOG}
PLAN_RANK = {"free": 0, "plus": 1, "pro": 2}
ACTIVE_SUBSCRIPTION_STATUSES = frozenset({"active", "trialing", "grace_period"})
TERMINAL_SUBSCRIPTION_STATUSES = frozenset(
    {
        "canceled",
        "cancelled",
        "expired",
        "incomplete_expired",
        "refunded",
        "revoked",
    }
)
PRODUCTION_ENVIRONMENT = "Production"
SANDBOX_ENVIRONMENT = "Sandbox"
UNKNOWN_ENVIRONMENT = "Unknown"

def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def billing_enforcement_enabled() -> bool:
    return _truthy(os.getenv("BILLING_ENFORCEMENT_ENABLED", "0"))


def normalize_provider_environment(value: object) -> str:
    raw_value = getattr(value, "value", value)
    normalized = str(raw_value or "").strip().lower().replace("_", "")
    if normalized in {"production", "live", "livemode"}:
        return PRODUCTION_ENVIRONMENT
    if normalized in {"sandbox", "test", "testmode"}:
        return SANDBOX_ENVIRONMENT
    return UNKNOWN_ENVIRONMENT


def billing_entitlement_environment() -> str:
    """Return the one provider environment allowed to grant this deployment."""
    configured = normalize_provider_environment(
        os.getenv("BILLING_ENTITLEMENT_ENVIRONMENT", PRODUCTION_ENVIRONMENT)
    )
    # A missing or misspelled value must never make sandbox rows live.
    return (
        configured
        if configured in {PRODUCTION_ENVIRONMENT, SANDBOX_ENVIRONMENT}
        else PRODUCTION_ENVIRONMENT
    )


def _utc_now(value: datetime | str | None = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | str | None = None) -> str:
    return _utc_now(value).isoformat()


def _day_and_reset(now: datetime | str | None = None) -> tuple[str, str]:
    current = _utc_now(now)
    usage_day = current.date().isoformat()
    reset = datetime.combine(
        current.date() + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    return usage_day, reset.isoformat()


def plans_payload() -> dict[str, Any]:
    return {"plans": [dict(plan) for plan in PLAN_CATALOG]}


def _stripe_price_catalog() -> dict[str, str]:
    prices = {
        "plus": os.getenv("STRIPE_PLUS_PRICE_ID", "").strip(),
        "pro": os.getenv("STRIPE_PRO_PRICE_ID", "").strip(),
    }
    missing = [
        f"STRIPE_{plan.upper()}_PRICE_ID"
        for plan, price_id in prices.items()
        if not price_id
    ]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} is not configured")
    if prices["plus"] == prices["pro"]:
        raise RuntimeError("STRIPE_PLUS_PRICE_ID and STRIPE_PRO_PRICE_ID must be different")
    return prices


def stripe_price_for_plan(plan_code: str) -> str:
    normalized_plan = str(plan_code or "").strip().lower()
    if normalized_plan not in {"plus", "pro"}:
        raise ValueError("plan must be plus or pro")
    return _stripe_price_catalog()[normalized_plan]


def plan_for_product(provider: str, product_id: object) -> str | None:
    normalized_provider = str(provider or "").strip().lower()
    normalized_product = str(product_id or "").strip()
    if normalized_provider == "stripe":
        mapping = {price_id: plan for plan, price_id in _stripe_price_catalog().items()}
        return mapping.get(normalized_product)
    return None


def _subscription_is_active(row: dict[str, Any], now: datetime) -> bool:
    if str(row.get("status") or "").strip().lower() not in ACTIVE_SUBSCRIPTION_STATUSES:
        return False
    raw_end = str(row.get("current_period_end") or "").strip()
    if not raw_end:
        return False
    try:
        return _utc_now(raw_end) > now
    except (TypeError, ValueError):
        return False


def subscription_rows(conn: Any, account_id: str) -> list[dict[str, Any]]:
    return fetch_all(
        conn,
        """
        SELECT provider, provider_environment, plan_code, status, current_period_end,
               cancel_at_period_end, external_product_id,
               external_subscription_id, provider_event_created_at
        FROM billing_subscriptions
        WHERE account_id = ? AND provider = 'stripe' AND provider_environment = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (account_id, billing_entitlement_environment()),
    )


def effective_plan(
    conn: Any,
    account_id: str,
    *,
    now: datetime | str | None = None,
) -> str:
    current = _utc_now(now)
    winner = "free"
    for row in subscription_rows(conn, account_id):
        plan_code = str(row.get("plan_code") or "").strip().lower()
        if (
            plan_code in PLAN_RANK
            and _subscription_is_active(row, current)
            and PLAN_RANK[plan_code] > PLAN_RANK[winner]
        ):
            winner = plan_code
    return winner


def billing_status(
    conn: Any,
    account_id: str,
    *,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    current = _utc_now(now)
    usage_day, reset_at = _day_and_reset(current)
    _acquire_quota_lock(conn, account_id, usage_day)
    reconcile_stale_reservations(conn, now=current, account_id=account_id)
    plan_code = effective_plan(conn, account_id, now=current)
    daily_limit = int(PLAN_BY_CODE[plan_code]["daily_limit"])
    usage = fetch_one(
        conn,
        "SELECT used_count FROM daily_search_usage WHERE account_id = ? AND usage_day = ?",
        (account_id, usage_day),
    )
    used = max(0, int((usage or {}).get("used_count") or 0))
    subscriptions = [
        {
            "provider": str(row.get("provider") or ""),
            "plan": str(row.get("plan_code") or "free"),
            "status": str(row.get("status") or "unknown"),
            "current_period_end": row.get("current_period_end"),
            "cancel_at_period_end": bool(row.get("cancel_at_period_end")),
            "product_id": str(row.get("external_product_id") or ""),
        }
        for row in subscription_rows(conn, account_id)
    ]
    return {
        "plan": plan_code,
        "daily_limit": daily_limit,
        "used_searches": used,
        "remaining_searches": max(0, daily_limit - used),
        "reset_at": reset_at,
        "subscriptions": subscriptions,
    }


@dataclass(frozen=True)
class DailySearchLimitReached(RuntimeError):
    plan: str
    limit: int
    used: int
    reset_at: str
    observed_at: str

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def retry_after_seconds(self) -> int:
        reset = _utc_now(self.reset_at)
        return max(1, int((reset - _utc_now(self.observed_at)).total_seconds()))

    def detail(self) -> dict[str, Any]:
        return {
            "code": "daily_search_limit_reached",
            "plan": self.plan,
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
        }


def _quota_lock_id(account_id: str, usage_day: str) -> int:
    digest = hashlib.sha256(f"{account_id}:{usage_day}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _acquire_quota_lock(conn: Any, account_id: str, usage_day: str) -> None:
    if isinstance(conn, sqlite3.Connection):
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        return
    fetch_one(
        conn,
        "SELECT pg_advisory_xact_lock(?) AS acquired",
        (_quota_lock_id(account_id, usage_day),),
    )


def _attach_reservation_job(
    conn: Any,
    *,
    reservation_id: str,
    generation_job_id: str,
    attached_at: str,
) -> None:
    upsert(
        conn,
        "search_quota_reservation_jobs",
        {
            "reservation_id": reservation_id,
            "generation_job_id": generation_job_id,
            "attached_at": attached_at,
        },
        pk=["reservation_id", "generation_job_id"],
    )


def reserve_search(
    conn: Any,
    *,
    account_id: str,
    operation_key: str,
    surface: str,
    material_id: str | None = None,
    generation_job_id: str | None = None,
    now: datetime | str | None = None,
    enforce: bool | None = None,
) -> dict[str, Any] | None:
    """Reserve one daily search; stable operation keys make retries free."""
    if enforce is None:
        enforce = billing_enforcement_enabled()
    if not enforce:
        return None
    clean_account = str(account_id or "").strip()
    clean_operation = str(operation_key or "").strip()
    clean_surface = str(surface or "").strip()
    if not clean_account or not clean_operation or not clean_surface:
        raise ValueError("account_id, operation_key, and surface are required")

    timestamp = _iso(now)
    usage_day, reset_at = _day_and_reset(timestamp)
    _acquire_quota_lock(conn, clean_account, usage_day)
    reconcile_stale_reservations(
        conn,
        now=timestamp,
        account_id=clean_account,
    )
    existing = fetch_one(
        conn,
        "SELECT * FROM search_quota_reservations WHERE account_id = ? AND operation_key = ?",
        (clean_account, clean_operation),
    )
    if existing and str(existing.get("status") or "") in {"reserved", "consumed"}:
        if generation_job_id:
            execute_modify(
                conn,
                """
                UPDATE search_quota_reservations
                SET generation_job_id = COALESCE(generation_job_id, ?),
                    material_id = COALESCE(?, material_id), updated_at = ?
                WHERE id = ?
                """,
                (generation_job_id, material_id, timestamp, existing["id"]),
            )
            _attach_reservation_job(
                conn,
                reservation_id=str(existing["id"]),
                generation_job_id=generation_job_id,
                attached_at=timestamp,
            )
            settle_job_reservation(conn, generation_job_id)
        return fetch_one(
            conn,
            "SELECT * FROM search_quota_reservations WHERE id = ?",
            (existing["id"],),
        ) or existing

    status = billing_status(conn, clean_account, now=timestamp)
    limit = int(status["daily_limit"])
    used = int(status["used_searches"])
    if used >= limit:
        raise DailySearchLimitReached(
            plan=str(status["plan"]),
            limit=limit,
            used=used,
            reset_at=reset_at,
            observed_at=timestamp,
        )

    usage = fetch_one(
        conn,
        "SELECT used_count FROM daily_search_usage WHERE account_id = ? AND usage_day = ?",
        (clean_account, usage_day),
    )
    if usage:
        execute_modify(
            conn,
            "UPDATE daily_search_usage SET used_count = used_count + 1, updated_at = ? WHERE account_id = ? AND usage_day = ?",
            (timestamp, clean_account, usage_day),
        )
    else:
        insert(
            conn,
            "daily_search_usage",
            {
                "account_id": clean_account,
                "usage_day": usage_day,
                "used_count": 1,
                "updated_at": timestamp,
            },
        )

    reservation_id = str((existing or {}).get("id") or uuid.uuid4())
    if existing:
        execute_modify(
            conn,
            """
            UPDATE search_quota_reservations
            SET usage_day = ?, surface = ?, plan_code = ?, material_id = ?, generation_job_id = ?,
                status = 'reserved', updated_at = ?, finalized_at = NULL
            WHERE id = ?
            """,
            (
                usage_day,
                clean_surface,
                str(status["plan"]),
                material_id,
                generation_job_id,
                timestamp,
                reservation_id,
            ),
        )
    else:
        insert(
            conn,
            "search_quota_reservations",
            {
                "id": reservation_id,
                "account_id": clean_account,
                "operation_key": clean_operation,
                "usage_day": usage_day,
                "surface": clean_surface,
                "plan_code": str(status["plan"]),
                "material_id": material_id,
                "generation_job_id": generation_job_id,
                "status": "reserved",
                "created_at": timestamp,
                "updated_at": timestamp,
                "finalized_at": None,
            },
        )
    reservation = fetch_one(
        conn,
        "SELECT * FROM search_quota_reservations WHERE id = ?",
        (reservation_id,),
    )
    if reservation and generation_job_id:
        _attach_reservation_job(
            conn,
            reservation_id=reservation_id,
            generation_job_id=generation_job_id,
            attached_at=timestamp,
        )
        settle_job_reservation(conn, generation_job_id)
    return reservation


def attach_reservation_to_job(
    conn: Any,
    *,
    account_id: str,
    operation_key: str,
    generation_job_id: str,
    material_id: str | None = None,
) -> None:
    reservation = fetch_one(
        conn,
        """
        SELECT id FROM search_quota_reservations
        WHERE account_id = ? AND operation_key = ?
          AND status IN ('reserved', 'consumed')
        """,
        (account_id, operation_key),
    )
    if not reservation:
        return
    timestamp = _iso()
    updated = execute_modify(
        conn,
        """
        UPDATE search_quota_reservations
        SET generation_job_id = COALESCE(generation_job_id, ?),
            material_id = COALESCE(?, material_id), updated_at = ?
        WHERE id = ? AND status IN ('reserved', 'consumed')
        """,
        (generation_job_id, material_id, timestamp, reservation["id"]),
    )
    if not updated:
        return
    _attach_reservation_job(
        conn,
        reservation_id=str(reservation["id"]),
        generation_job_id=generation_job_id,
        attached_at=timestamp,
    )
    settle_job_reservation(conn, generation_job_id)


def settle_reservation(
    conn: Any,
    *,
    reservation_id: str,
    usable_result: bool,
    now: datetime | str | None = None,
) -> bool:
    timestamp = _iso(now)
    row = fetch_one(
        conn,
        "SELECT * FROM search_quota_reservations WHERE id = ?",
        (reservation_id,),
    )
    if not row:
        return False
    _acquire_quota_lock(conn, str(row["account_id"]), str(row["usage_day"]))
    row = fetch_one(
        conn,
        "SELECT * FROM search_quota_reservations WHERE id = ?",
        (reservation_id,),
    )
    if not row or str(row.get("status") or "") != "reserved":
        return False
    if usable_result:
        return bool(
            execute_modify(
                conn,
                "UPDATE search_quota_reservations SET status = 'consumed', updated_at = ?, finalized_at = ? WHERE id = ? AND status = 'reserved'",
                (timestamp, timestamp, reservation_id),
            )
        )
    updated = execute_modify(
        conn,
        "UPDATE search_quota_reservations SET status = 'refunded', updated_at = ?, finalized_at = ? WHERE id = ? AND status = 'reserved'",
        (timestamp, timestamp, reservation_id),
    )
    if updated:
        execute_modify(
            conn,
            """
            UPDATE daily_search_usage
            SET used_count = CASE WHEN used_count > 0 THEN used_count - 1 ELSE 0 END,
                updated_at = ?
            WHERE account_id = ? AND usage_day = ?
            """,
            (timestamp, row["account_id"], row["usage_day"]),
        )
    return bool(updated)


def settle_operation(
    conn: Any,
    *,
    account_id: str,
    operation_key: str,
    usable_result: bool,
    now: datetime | str | None = None,
) -> bool:
    row = fetch_one(
        conn,
        "SELECT id FROM search_quota_reservations WHERE account_id = ? AND operation_key = ?",
        (account_id, operation_key),
    )
    return bool(
        row
        and settle_reservation(
            conn,
            reservation_id=str(row["id"]),
            usable_result=usable_result,
            now=now,
        )
    )


def reconcile_stale_reservations(
    conn: Any,
    *,
    now: datetime | str | None = None,
    max_age_seconds: int = 2 * 60 * 60,
    account_id: str | None = None,
) -> int:
    """Refund reservations whose UTC day or maximum job lifetime has elapsed."""
    current = _utc_now(now)
    usage_day, _reset_at = _day_and_reset(current)
    cutoff = (current - timedelta(seconds=max(60, int(max_age_seconds)))).isoformat()
    account_clause = " AND account_id = ?" if account_id else ""
    params: tuple[Any, ...] = (
        (usage_day, cutoff, account_id)
        if account_id
        else (usage_day, cutoff)
    )
    rows = fetch_all(
        conn,
        f"""
        SELECT id FROM search_quota_reservations
        WHERE status = 'reserved'
          AND (usage_day < ? OR updated_at <= ?)
          {account_clause}
        """,
        params,
    )
    refunded = 0
    for row in rows:
        if settle_reservation(
            conn,
            reservation_id=str(row["id"]),
            usable_result=False,
            now=current,
        ):
            refunded += 1
    return refunded


def _job_has_usable_result(conn: Any, job_row: dict[str, Any]) -> bool:
    status = str(job_row.get("status") or "")
    if status not in {"completed", "partial"}:
        return False
    final_event = fetch_one(
        conn,
        """
        SELECT payload_json FROM generation_job_events
        WHERE job_id = ? AND event_type = 'final'
        ORDER BY seq DESC LIMIT 1
        """,
        (job_row["id"],),
    )
    if final_event:
        import json

        try:
            payload = json.loads(str(final_event.get("payload_json") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("result"):
            return True
        if isinstance(payload.get("reels"), list) and bool(payload["reels"]):
            return True
        if int(payload.get("batch_size") or 0) > 0:
            return True
    generation_id = str(job_row.get("result_generation_id") or "").strip()
    if generation_id:
        count = fetch_one(
            conn,
            "SELECT COUNT(*) AS total FROM reels WHERE generation_id = ?",
            (generation_id,),
        )
        return int((count or {}).get("total") or 0) > 0
    return False


def settle_job_reservation(conn: Any, job_id: str) -> bool:
    reservations = fetch_all(
        conn,
        """
        SELECT DISTINCT r.id, r.account_id, r.usage_day
        FROM search_quota_reservations AS r
        LEFT JOIN search_quota_reservation_jobs AS j ON j.reservation_id = r.id
        WHERE r.status = 'reserved'
          AND (j.generation_job_id = ? OR r.generation_job_id = ?)
        """,
        (job_id, job_id),
    )
    if not reservations:
        return False
    reservations.sort(
        key=lambda row: (
            _quota_lock_id(str(row["account_id"]), str(row["usage_day"])),
            str(row["id"]),
        )
    )
    for reservation in reservations:
        _acquire_quota_lock(
            conn,
            str(reservation["account_id"]),
            str(reservation["usage_day"]),
        )
    terminal_statuses = {
        "completed",
        "partial",
        "exhausted",
        "failed",
        "cancelled",
    }
    settled = False
    for reservation in reservations:
        reservation_id = str(reservation["id"])
        current = fetch_one(
            conn,
            "SELECT status FROM search_quota_reservations WHERE id = ?",
            (reservation_id,),
        )
        if not current or str(current.get("status") or "") != "reserved":
            continue
        jobs = fetch_all(
            conn,
            """
            SELECT g.*
            FROM reel_generation_jobs AS g
            JOIN search_quota_reservation_jobs AS j
              ON j.generation_job_id = g.id
            WHERE j.reservation_id = ?
            """,
            (reservation_id,),
        )
        if not jobs:
            legacy_job = fetch_one(
                conn,
                """
                SELECT g.* FROM reel_generation_jobs AS g
                JOIN search_quota_reservations AS r ON r.generation_job_id = g.id
                WHERE r.id = ?
                """,
                (reservation_id,),
            )
            jobs = [legacy_job] if legacy_job else []
        usable_result = any(_job_has_usable_result(conn, job) for job in jobs)
        all_terminal = bool(jobs) and all(
            str(job.get("status") or "") in terminal_statuses for job in jobs
        )
        if usable_result or all_terminal:
            settled = (
                settle_reservation(
                    conn,
                    reservation_id=reservation_id,
                    usable_result=usable_result,
                )
                or settled
            )
    return settled


def upsert_provider_customer(
    conn: Any,
    *,
    account_id: str,
    provider: str,
    provider_environment: str,
    external_customer_id: str,
    now: datetime | str | None = None,
) -> None:
    timestamp = _iso(now)
    normalized_environment = normalize_provider_environment(provider_environment)
    if normalized_environment not in {PRODUCTION_ENVIRONMENT, SANDBOX_ENVIRONMENT}:
        raise ValueError("provider_environment must be Production or Sandbox")
    bound_customer = fetch_one(
        conn,
        "SELECT account_id FROM billing_provider_customers "
        "WHERE provider = ? AND provider_environment = ? "
        "AND external_customer_id = ?",
        (provider, normalized_environment, external_customer_id),
    )
    if bound_customer and str(bound_customer.get("account_id") or "") != account_id:
        raise ValueError(
            "Provider customer is already bound to another ReelAI account."
        )
    existing = fetch_one(
        conn,
        "SELECT account_id FROM billing_provider_customers "
        "WHERE account_id = ? AND provider = ? AND provider_environment = ?",
        (account_id, provider, normalized_environment),
    )
    if existing:
        execute_modify(
            conn,
            "UPDATE billing_provider_customers SET external_customer_id = ?, "
            "updated_at = ? WHERE account_id = ? AND provider = ? "
            "AND provider_environment = ?",
            (
                external_customer_id,
                timestamp,
                account_id,
                provider,
                normalized_environment,
            ),
        )
    else:
        insert(
            conn,
            "billing_provider_customers",
            {
                "account_id": account_id,
                "provider": provider,
                "provider_environment": normalized_environment,
                "external_customer_id": external_customer_id,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )


def provider_customer_id(
    conn: Any,
    account_id: str,
    provider: str,
    *,
    provider_environment: str | None = None,
) -> str | None:
    environment = normalize_provider_environment(
        provider_environment or billing_entitlement_environment()
    )
    row = fetch_one(
        conn,
        "SELECT external_customer_id FROM billing_provider_customers "
        "WHERE account_id = ? AND provider = ? AND provider_environment = ?",
        (account_id, provider, environment),
    )
    return str((row or {}).get("external_customer_id") or "").strip() or None


def upsert_subscription(
    conn: Any,
    *,
    account_id: str,
    provider: str,
    external_subscription_id: str,
    external_product_id: str,
    plan_code: str,
    status: str,
    current_period_end: datetime | str | None,
    provider_environment: str,
    cancel_at_period_end: bool = False,
    provider_event_created_at: datetime | str | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    if plan_code not in {"plus", "pro"}:
        raise ValueError("plan_code must be plus or pro")
    normalized_environment = normalize_provider_environment(provider_environment)
    if normalized_environment not in {PRODUCTION_ENVIRONMENT, SANDBOX_ENVIRONMENT}:
        raise ValueError("provider_environment must be Production or Sandbox")
    timestamp = _iso(now)
    event_time = _iso(provider_event_created_at or timestamp)
    normalized_status = str(status or "unknown").strip().lower()
    existing = fetch_one(
        conn,
        "SELECT * FROM billing_subscriptions WHERE provider = ? AND external_subscription_id = ?",
        (provider, external_subscription_id),
    )
    if existing:
        existing_account = str(existing.get("account_id") or "")
        existing_environment = normalize_provider_environment(
            existing.get("provider_environment")
        )
        if existing_account != account_id:
            raise ValueError(
                "Provider subscription is already bound to another ReelAI account."
            )
        if existing_environment != normalized_environment:
            raise ValueError(
                "Provider subscription cannot change billing environment."
            )
    subscription_id = str((existing or {}).get("id") or uuid.uuid4())
    terminal_statuses = (
        "'canceled', 'cancelled', 'expired', "
        "'incomplete_expired', 'refunded', 'revoked'"
    )
    excluded_plan_rank = (
        "CASE excluded.plan_code WHEN 'pro' THEN 2 WHEN 'plus' THEN 1 ELSE 0 END"
    )
    stored_plan_rank = (
        "CASE billing_subscriptions.plan_code "
        "WHEN 'pro' THEN 2 WHEN 'plus' THEN 1 ELSE 0 END"
    )
    equal_state_tiebreaker = """
        excluded.cancel_at_period_end > billing_subscriptions.cancel_at_period_end
        OR (
            excluded.cancel_at_period_end = billing_subscriptions.cancel_at_period_end
            AND COALESCE(excluded.current_period_end, '')
                > COALESCE(billing_subscriptions.current_period_end, '')
        )
        OR (
            excluded.cancel_at_period_end = billing_subscriptions.cancel_at_period_end
            AND COALESCE(excluded.current_period_end, '')
                = COALESCE(billing_subscriptions.current_period_end, '')
            AND excluded.external_product_id
                > billing_subscriptions.external_product_id
        )
    """
    execute_modify(
        conn,
        f"""
        INSERT INTO billing_subscriptions (
            id, account_id, provider, provider_environment,
            external_subscription_id, external_product_id, plan_code, status,
            current_period_end, cancel_at_period_end, provider_event_created_at,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, external_subscription_id) DO UPDATE SET
            external_product_id = excluded.external_product_id,
            plan_code = excluded.plan_code,
            status = excluded.status,
            current_period_end = excluded.current_period_end,
            cancel_at_period_end = excluded.cancel_at_period_end,
            provider_event_created_at = excluded.provider_event_created_at,
            updated_at = excluded.updated_at
        WHERE billing_subscriptions.account_id = excluded.account_id
          AND billing_subscriptions.provider_environment = excluded.provider_environment
          AND (
              billing_subscriptions.provider_event_created_at IS NULL
              OR billing_subscriptions.provider_event_created_at < excluded.provider_event_created_at
              OR (
                  billing_subscriptions.provider_event_created_at = excluded.provider_event_created_at
                  AND (
                      (
                          billing_subscriptions.status NOT IN ({terminal_statuses})
                          AND excluded.status IN ({terminal_statuses})
                      )
                      OR (
                          billing_subscriptions.status IN ({terminal_statuses})
                          AND excluded.status IN ({terminal_statuses})
                          AND (
                              excluded.status > billing_subscriptions.status
                              OR (
                                  excluded.status = billing_subscriptions.status
                                  AND ({equal_state_tiebreaker})
                              )
                          )
                      )
                      OR (
                          billing_subscriptions.status NOT IN ({terminal_statuses})
                          AND excluded.status NOT IN ({terminal_statuses})
                          AND (
                              {excluded_plan_rank} > {stored_plan_rank}
                              OR (
                                  {excluded_plan_rank} = {stored_plan_rank}
                                  AND (
                                      excluded.status > billing_subscriptions.status
                                      OR (
                                          excluded.status = billing_subscriptions.status
                                          AND ({equal_state_tiebreaker})
                                      )
                                  )
                              )
                          )
                      )
                  )
              )
          )
        """,
        (
            subscription_id,
            account_id,
            provider,
            normalized_environment,
            external_subscription_id,
            external_product_id,
            plan_code,
            normalized_status,
            _iso(current_period_end) if current_period_end else None,
            1 if cancel_at_period_end else 0,
            event_time,
            timestamp,
            timestamp,
        ),
    )
    stored = fetch_one(
        conn,
        "SELECT * FROM billing_subscriptions WHERE provider = ? AND external_subscription_id = ?",
        (provider, external_subscription_id),
    ) or {}
    if str(stored.get("account_id") or "") != account_id:
        raise ValueError(
            "Provider subscription is already bound to another ReelAI account."
        )
    if normalize_provider_environment(stored.get("provider_environment")) != normalized_environment:
        raise ValueError("Provider subscription cannot change billing environment.")
    return stored


def record_provider_event(
    conn: Any,
    *,
    provider: str,
    external_event_id: str,
    event_type: str,
    external_event_created_at: datetime | str | None = None,
    now: datetime | str | None = None,
) -> bool:
    return bool(
        execute_modify(
            conn,
            """
            INSERT INTO billing_provider_events (
                provider, external_event_id, external_event_created_at,
                event_type, processed_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_event_id) DO NOTHING
            """,
            (
                provider,
                external_event_id,
                _iso(external_event_created_at) if external_event_created_at else None,
                event_type,
                _iso(now),
            ),
        )
    )


__all__ = [
    "DailySearchLimitReached",
    "PRODUCTION_ENVIRONMENT",
    "SANDBOX_ENVIRONMENT",
    "TERMINAL_SUBSCRIPTION_STATUSES",
    "attach_reservation_to_job",
    "billing_enforcement_enabled",
    "billing_entitlement_environment",
    "billing_status",
    "effective_plan",
    "normalize_provider_environment",
    "plan_for_product",
    "plans_payload",
    "provider_customer_id",
    "record_provider_event",
    "reconcile_stale_reservations",
    "reserve_search",
    "settle_job_reservation",
    "settle_operation",
    "settle_reservation",
    "stripe_price_for_plan",
    "subscription_rows",
    "upsert_provider_customer",
    "upsert_subscription",
]
