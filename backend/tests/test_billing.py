from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend.app import db
from backend.app.services import billing
from backend.app.services import billing_providers
from backend.app.services import generation_jobs


BASE_TIME = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _conn(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(db.SCHEMA)
    return conn


def _account(conn: sqlite3.Connection, account_id: str | None = None) -> str:
    value = account_id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO community_accounts (
            id, username, username_normalized, email, email_normalized,
            password_hash, password_salt, verified_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            value,
            f"user-{value[:8]}",
            f"user-{value[:8]}",
            f"{value[:8]}@example.com",
            f"{value[:8]}@example.com",
            "hash",
            "salt",
            BASE_TIME.isoformat(),
            BASE_TIME.isoformat(),
            BASE_TIME.isoformat(),
        ),
    )
    return value


def _subscribe(
    conn: sqlite3.Connection,
    account_id: str,
    *,
    provider: str,
    plan: str,
    event_time: datetime = BASE_TIME,
    status: str = "active",
    provider_environment: str = "Production",
) -> None:
    billing.upsert_subscription(
        conn,
        account_id=account_id,
        provider=provider,
        external_subscription_id=f"{provider}-{plan}-{account_id}",
        external_product_id=f"{provider}-{plan}",
        plan_code=plan,
        status=status,
        current_period_end=BASE_TIME + timedelta(days=30),
        provider_environment=provider_environment,
        provider_event_created_at=event_time,
        now=event_time,
    )


@contextmanager
def _fixed_connection(conn: sqlite3.Connection, *, transactional: bool = False):
    if transactional and not conn.in_transaction:
        conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    else:
        if conn.in_transaction:
            conn.commit()


def _authenticated_material(conn: sqlite3.Connection) -> tuple[str, str, dict[str, str]]:
    account_id = _account(conn)
    material_id = f"material-{account_id}"
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) "
        "VALUES (?, ?, ?, ?)",
        (material_id, "Cellular respiration", "topic", BASE_TIME.isoformat()),
    )
    session_token = f"session-{account_id}"
    conn.execute(
        "INSERT INTO community_sessions "
        "(id, account_id, token_hash, created_at, last_used_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            account_id,
            hashlib.sha256(session_token.encode("utf-8")).hexdigest(),
            BASE_TIME.isoformat(),
            BASE_TIME.isoformat(),
            (BASE_TIME + timedelta(days=365)).isoformat(),
        ),
    )
    return account_id, material_id, {
        "x-studyreels-owner-key": f"owner-{account_id}",
        "x-studyreels-session-token": session_token,
    }


def test_schema_and_public_plan_contract_default_existing_accounts_to_free() -> None:
    conn = _conn()
    account_id = _account(conn)

    assert billing.plans_payload() == {
        "plans": [
            {"code": "free", "name": "Free", "monthly_price_cents": 0, "daily_limit": 5},
            {"code": "plus", "name": "Plus", "monthly_price_cents": 499, "daily_limit": 15},
            {"code": "pro", "name": "Pro", "monthly_price_cents": 1999, "daily_limit": 50},
        ]
    }
    status = billing.billing_status(conn, account_id, now=BASE_TIME)
    assert status == {
        "plan": "free",
        "daily_limit": 5,
        "used_searches": 0,
        "remaining_searches": 5,
        "reset_at": "2026-07-19T00:00:00+00:00",
        "subscriptions": [],
    }


def test_reservation_job_migration_backfills_the_legacy_link() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-migration", "Biology", "topic", BASE_TIME.isoformat()),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-migration",
        concept_id=None,
        request_key="migration-request",
        content_fingerprint="migration-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:migration",
        surface="generation",
        generation_job_id=job["id"],
        now=BASE_TIME,
        enforce=True,
    )
    conn.execute(
        "DELETE FROM search_quota_reservation_jobs WHERE reservation_id = ?",
        (reservation["id"],),
    )

    db._migrate_search_quota_reservation_jobs_sqlite(conn)

    linked = conn.execute(
        "SELECT generation_job_id FROM search_quota_reservation_jobs "
        "WHERE reservation_id = ?",
        (reservation["id"],),
    ).fetchone()
    assert linked[0] == job["id"]


def test_operation_replay_is_free_and_refunded_or_stale_work_can_reopen() -> None:
    conn = _conn()
    account_id = _account(conn)
    first = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:stable",
        surface="test",
        now=BASE_TIME,
        enforce=True,
    )
    replay = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:stable",
        surface="test",
        now=BASE_TIME,
        enforce=True,
    )
    assert replay["id"] == first["id"]
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 1

    assert billing.settle_operation(
        conn,
        account_id=account_id,
        operation_key="material:stable",
        usable_result=False,
        now=BASE_TIME,
    )
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 0
    reopened = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:stable",
        surface="test",
        now=BASE_TIME + timedelta(minutes=1),
        enforce=True,
    )
    assert reopened["status"] == "reserved"

    stale_now = BASE_TIME + timedelta(hours=3)
    reopened_again = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:stable",
        surface="test",
        now=stale_now,
        enforce=True,
    )
    assert reopened_again["status"] == "reserved"
    assert billing.billing_status(conn, account_id, now=stale_now)["used_searches"] == 1


def test_concurrent_sqlite_reservations_never_exceed_free_limit(tmp_path) -> None:
    path = str(tmp_path / "quota.sqlite3")
    setup = _conn(path)
    account_id = _account(setup)
    setup.close()
    barrier = threading.Barrier(10)

    def reserve(index: int) -> str:
        conn = _conn(path)
        try:
            barrier.wait(timeout=5)
            billing.reserve_search(
                conn,
                account_id=account_id,
                operation_key=f"material:{index}",
                surface="concurrent",
                now=BASE_TIME,
                enforce=True,
            )
            conn.commit()
            return "reserved"
        except billing.DailySearchLimitReached:
            conn.rollback()
            return "limited"
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(reserve, range(10)))
    assert outcomes.count("reserved") == 5
    assert outcomes.count("limited") == 5
    verify = _conn(path)
    assert billing.billing_status(verify, account_id, now=BASE_TIME)["used_searches"] == 5


def test_account_isolation_and_success_failure_settlement() -> None:
    conn = _conn()
    first_account = _account(conn)
    second_account = _account(conn)
    first = billing.reserve_search(
        conn,
        account_id=first_account,
        operation_key="material:same",
        surface="test",
        now=BASE_TIME,
        enforce=True,
    )
    second = billing.reserve_search(
        conn,
        account_id=second_account,
        operation_key="material:same",
        surface="test",
        now=BASE_TIME,
        enforce=True,
    )
    assert first["id"] != second["id"]
    assert billing.settle_reservation(
        conn, reservation_id=first["id"], usable_result=True, now=BASE_TIME
    )
    assert billing.settle_reservation(
        conn, reservation_id=second["id"], usable_result=False, now=BASE_TIME
    )
    assert billing.billing_status(conn, first_account, now=BASE_TIME)["used_searches"] == 1
    assert billing.billing_status(conn, second_account, now=BASE_TIME)["used_searches"] == 0


def test_generation_terminal_paths_consume_usable_results_and_refund_failures() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-1", "Biology", "topic", BASE_TIME.isoformat()),
    )

    successful_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-1",
        concept_id=None,
        request_key="successful-request",
        content_fingerprint="fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    successful_reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:successful",
        surface="generation",
        generation_job_id=successful_job["id"],
        now=BASE_TIME,
        enforce=True,
    )
    generation_jobs.append_event(
        conn,
        job_id=successful_job["id"],
        event_type="final",
        payload={"reels": [{"reel_id": "reel-1"}], "batch_size": 1},
        now=BASE_TIME,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=successful_job["id"],
        status="completed",
        now=BASE_TIME,
    )
    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (successful_reservation["id"],),
    ).fetchone()[0] == "consumed"

    failed_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-1",
        concept_id=None,
        request_key="failed-request",
        content_fingerprint="fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    failed_reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:failed",
        surface="generation",
        generation_job_id=failed_job["id"],
        now=BASE_TIME,
        enforce=True,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=failed_job["id"],
        status="failed",
        error_code="provider_failed",
        now=BASE_TIME,
    )
    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (failed_reservation["id"],),
    ).fetchone()[0] == "refunded"
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 1


def test_shared_material_reservation_waits_for_all_jobs_and_consumes_any_success() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-shared", "Biology", "topic", BASE_TIME.isoformat()),
    )
    conn.executemany(
        """
        INSERT INTO concepts (
            id, material_id, title, keywords_json, summary, created_at
        ) VALUES (?, 'material-shared', ?, '[]', ?, ?)
        """,
        [
            ("concept-a", "Concept A", "First concept", BASE_TIME.isoformat()),
            ("concept-b", "Concept B", "Second concept", BASE_TIME.isoformat()),
        ],
    )
    failed_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-shared",
        concept_id="concept-a",
        request_key="shared-failed-request",
        content_fingerprint="shared-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    successful_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-shared",
        concept_id="concept-b",
        request_key="shared-successful-request",
        content_fingerprint="shared-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:material-shared",
        surface="generation",
        generation_job_id=failed_job["id"],
        now=BASE_TIME,
        enforce=True,
    )
    billing.attach_reservation_to_job(
        conn,
        account_id=account_id,
        operation_key="material:material-shared",
        generation_job_id=successful_job["id"],
        material_id="material-shared",
    )

    generation_jobs.transition_terminal(
        conn,
        job_id=failed_job["id"],
        status="failed",
        error_code="provider_failed",
        now=BASE_TIME,
    )
    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (reservation["id"],),
    ).fetchone()[0] == "reserved"

    generation_jobs.append_event(
        conn,
        job_id=successful_job["id"],
        event_type="final",
        payload={"reels": [{"reel_id": "reel-shared"}], "batch_size": 1},
        now=BASE_TIME,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=successful_job["id"],
        status="completed",
        now=BASE_TIME,
    )
    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (reservation["id"],),
    ).fetchone()[0] == "consumed"
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 1


def test_shared_material_reservation_stays_consumed_when_success_finishes_first() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-success-first", "Biology", "topic", BASE_TIME.isoformat()),
    )
    successful_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-success-first",
        concept_id=None,
        request_key="success-first-request",
        content_fingerprint="shared-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    failed_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-success-first",
        concept_id=None,
        request_key="failure-second-request",
        content_fingerprint="shared-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:material-success-first",
        surface="generation",
        generation_job_id=successful_job["id"],
        now=BASE_TIME,
        enforce=True,
    )
    billing.attach_reservation_to_job(
        conn,
        account_id=account_id,
        operation_key="material:material-success-first",
        generation_job_id=failed_job["id"],
    )
    generation_jobs.append_event(
        conn,
        job_id=successful_job["id"],
        event_type="final",
        payload={"reels": [{"reel_id": "reel-success-first"}], "batch_size": 1},
        now=BASE_TIME,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=successful_job["id"],
        status="completed",
        now=BASE_TIME,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=failed_job["id"],
        status="failed",
        error_code="provider_failed",
        now=BASE_TIME,
    )

    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (reservation["id"],),
    ).fetchone()[0] == "consumed"
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 1


def test_terminal_job_is_settled_when_the_reservation_attaches_late() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-late-attach", "Biology", "topic", BASE_TIME.isoformat()),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-late-attach",
        concept_id=None,
        request_key="late-attach-request",
        content_fingerprint="late-attach-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:late-attach",
        surface="generation",
        now=BASE_TIME,
        enforce=True,
    )
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="final",
        payload={"reels": [{"reel_id": "late-result"}]},
        now=BASE_TIME,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=job["id"],
        status="completed",
        now=BASE_TIME,
    )
    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (reservation["id"],),
    ).fetchone()[0] == "reserved"

    billing.attach_reservation_to_job(
        conn,
        account_id=account_id,
        operation_key="material:late-attach",
        generation_job_id=job["id"],
        material_id="material-late-attach",
    )

    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (reservation["id"],),
    ).fetchone()[0] == "consumed"


@pytest.mark.parametrize("empty_result", [[], {}, ""])
def test_empty_final_result_refunds_the_search(empty_result) -> None:
    conn = _conn()
    account_id = _account(conn)
    material_id = f"material-empty-{type(empty_result).__name__}"
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        (material_id, "Biology", "topic", BASE_TIME.isoformat()),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id=material_id,
        concept_id=None,
        request_key=f"empty-{type(empty_result).__name__}",
        content_fingerprint="empty-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key=f"material:{material_id}",
        surface="generation",
        generation_job_id=job["id"],
        now=BASE_TIME,
        enforce=True,
    )
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="final",
        payload={"result": empty_result},
        now=BASE_TIME,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=job["id"],
        status="completed",
        now=BASE_TIME,
    )

    assert conn.execute(
        "SELECT status FROM search_quota_reservations WHERE id = ?",
        (reservation["id"],),
    ).fetchone()[0] == "refunded"


def test_job_settlement_locks_the_quota_before_reading_associated_jobs(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-lock-order", "Biology", "topic", BASE_TIME.isoformat()),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-lock-order",
        concept_id=None,
        request_key="lock-order-request",
        content_fingerprint="lock-order-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:lock-order",
        surface="generation",
        generation_job_id=job["id"],
        now=BASE_TIME,
        enforce=True,
    )
    events: list[str] = []
    real_fetch_all = billing.fetch_all

    def note_lock(*_args, **_kwargs):
        events.append("lock")

    def note_job_read(connection, query, params=()):
        if "SELECT g.*" in query:
            events.append("jobs")
        return real_fetch_all(connection, query, params)

    monkeypatch.setattr(billing, "_acquire_quota_lock", note_lock)
    monkeypatch.setattr(billing, "fetch_all", note_job_read)
    billing.settle_job_reservation(conn, job["id"])

    assert events.index("lock") < events.index("jobs")


def test_consumed_material_reservation_attaches_to_first_generation_job_for_costs() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-cost", "Biology", "topic", BASE_TIME.isoformat()),
    )
    first_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-cost",
        concept_id=None,
        request_key="first-cost-request",
        content_fingerprint="first-cost-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:material-cost",
        surface="material",
        material_id="material-cost",
        now=BASE_TIME,
        enforce=True,
    )
    assert billing.settle_reservation(
        conn,
        reservation_id=reservation["id"],
        usable_result=True,
        now=BASE_TIME,
    )

    billing.attach_reservation_to_job(
        conn,
        account_id=account_id,
        operation_key="material:material-cost",
        generation_job_id=first_job["id"],
        material_id="material-cost",
    )
    generation_jobs.record_provider_usage(
        conn,
        job_id=first_job["id"],
        provider="gemini",
        operation="generate",
        billable_requests=1,
        now=BASE_TIME,
    )

    linked = conn.execute(
        """
        SELECT r.status, r.plan_code, r.generation_job_id, u.billable_requests
        FROM search_quota_reservations AS r
        JOIN generation_provider_usage AS u ON u.job_id = r.generation_job_id
        WHERE r.id = ?
        """,
        (reservation["id"],),
    ).fetchone()
    assert tuple(linked) == ("consumed", "free", first_job["id"], 1)

    second_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="material-cost",
        concept_id=None,
        request_key="second-cost-request",
        content_fingerprint="second-cost-fingerprint",
        learner_id=account_id,
        request_params={},
        now=BASE_TIME,
    )
    billing.attach_reservation_to_job(
        conn,
        account_id=account_id,
        operation_key="material:material-cost",
        generation_job_id=second_job["id"],
        material_id="material-cost",
    )
    assert conn.execute(
        "SELECT generation_job_id FROM search_quota_reservations WHERE id = ?",
        (reservation["id"],),
    ).fetchone()[0] == first_job["id"]
    assert conn.execute(
        "SELECT COUNT(*) FROM search_quota_reservation_jobs WHERE reservation_id = ?",
        (reservation["id"],),
    ).fetchone()[0] == 2
    assert not billing.settle_reservation(
        conn,
        reservation_id=reservation["id"],
        usable_result=False,
        now=BASE_TIME,
    )
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 1


def test_stripe_events_are_replay_safe_and_ignore_out_of_order_state(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    fake_stripe = SimpleNamespace()
    monkeypatch.setattr(billing_providers, "_stripe_module", lambda: fake_stripe)

    def event(event_id: str, created: int, status: str, price: str) -> dict:
        return {
            "id": event_id,
            "type": "customer.subscription.updated",
            "created": created,
            "livemode": True,
            "data": {
                "object": {
                    "id": "sub_1",
                    "customer": "cus_1",
                    "metadata": {"account_id": account_id},
                    "items": {
                        "data": [
                            {
                                "price": {"id": price},
                                "current_period_end": int(
                                    (BASE_TIME + timedelta(days=30)).timestamp()
                                ),
                            }
                        ]
                    },
                    "status": status,
                    "cancel_at_period_end": False,
                }
            },
        }

    assert billing_providers.process_stripe_event(conn, event("evt_new", 200, "active", "price_pro"))
    assert not billing_providers.process_stripe_event(conn, event("evt_new", 200, "active", "price_pro"))
    assert billing_providers.process_stripe_event(conn, event("evt_old", 100, "canceled", "price_plus"))
    status = billing.billing_status(conn, account_id, now=BASE_TIME)
    assert status["plan"] == "pro"
    assert status["subscriptions"][0]["current_period_end"] == (
        BASE_TIME + timedelta(days=30)
    ).isoformat()


def test_postgres_stripe_event_lock_is_acquired_before_dedupe(monkeypatch) -> None:
    queries: list[str] = []

    def fake_fetch_one(_conn, query, _params=()):
        queries.append(query)
        if "billing_provider_events" in query:
            return {"external_event_id": "evt_duplicate"}
        return {"acquired": None}

    monkeypatch.setattr(billing_providers, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(),
    )

    processed = billing_providers.process_stripe_event(
        object(),
        {
            "id": "evt_duplicate",
            "type": "customer.subscription.updated",
            "created": 100,
        },
    )

    assert not processed
    assert "pg_advisory_xact_lock" in queries[0]
    assert "billing_provider_events" in queries[1]


def test_subscription_upsert_rechecks_event_order_in_the_atomic_write(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    _subscribe(
        conn,
        account_id,
        provider="stripe",
        plan="plus",
        event_time=BASE_TIME,
    )
    stale_snapshot = dict(
        conn.execute(
            "SELECT * FROM billing_subscriptions WHERE account_id = ?",
            (account_id,),
        ).fetchone()
    )
    subscription_id = str(stale_snapshot["external_subscription_id"])
    newer_time = BASE_TIME + timedelta(seconds=2)
    billing.upsert_subscription(
        conn,
        account_id=account_id,
        provider="stripe",
        external_subscription_id=subscription_id,
        external_product_id="stripe-pro",
        plan_code="pro",
        status="canceled",
        current_period_end=BASE_TIME + timedelta(days=30),
        provider_environment="Production",
        provider_event_created_at=newer_time,
        now=newer_time,
    )

    real_fetch_one = billing.fetch_one
    stale_returned = False

    def stale_initial_read(connection, query, params=()):
        nonlocal stale_returned
        if (
            not stale_returned
            and "SELECT * FROM billing_subscriptions WHERE provider = ?" in query
        ):
            stale_returned = True
            return dict(stale_snapshot)
        return real_fetch_one(connection, query, params)

    monkeypatch.setattr(billing, "fetch_one", stale_initial_read)
    stored = billing.upsert_subscription(
        conn,
        account_id=account_id,
        provider="stripe",
        external_subscription_id=subscription_id,
        external_product_id="stripe-plus",
        plan_code="plus",
        status="active",
        current_period_end=BASE_TIME + timedelta(days=30),
        provider_environment="Production",
        provider_event_created_at=BASE_TIME + timedelta(seconds=1),
        now=BASE_TIME + timedelta(seconds=3),
    )

    assert stored["status"] == "canceled"
    assert stored["plan_code"] == "pro"
    assert stored["provider_event_created_at"] == newer_time.isoformat()


def test_equal_timestamp_subscription_updates_choose_the_highest_plan() -> None:
    conn = _conn()
    account_id = _account(conn)
    _subscribe(conn, account_id, provider="stripe", plan="plus")
    subscription_id = f"stripe-plus-{account_id}"
    pro = billing.upsert_subscription(
        conn,
        account_id=account_id,
        provider="stripe",
        external_subscription_id=subscription_id,
        external_product_id="stripe-pro",
        plan_code="pro",
        status="active",
        current_period_end=BASE_TIME + timedelta(days=30),
        provider_environment="Production",
        provider_event_created_at=BASE_TIME,
        now=BASE_TIME,
    )
    plus = billing.upsert_subscription(
        conn,
        account_id=account_id,
        provider="stripe",
        external_subscription_id=subscription_id,
        external_product_id="stripe-plus",
        plan_code="plus",
        status="active",
        current_period_end=BASE_TIME + timedelta(days=30),
        provider_environment="Production",
        provider_event_created_at=BASE_TIME,
        now=BASE_TIME,
    )

    assert pro["plan_code"] == "pro"
    assert plus["plan_code"] == "pro"


def test_equal_timestamp_equal_state_uses_a_stable_cancellation_tiebreaker() -> None:
    def final_state(cancel_first: bool) -> tuple[int, str]:
        conn = _conn()
        account_id = _account(conn)
        subscription_id = f"stripe-plus-tie-{account_id}"
        states = [cancel_first, not cancel_first]
        for cancel_at_period_end in states:
            billing.upsert_subscription(
                conn,
                account_id=account_id,
                provider="stripe",
                external_subscription_id=subscription_id,
                external_product_id="stripe-plus",
                plan_code="plus",
                status="active",
                current_period_end=(
                    BASE_TIME + timedelta(days=20 if cancel_at_period_end else 30)
                ),
                provider_environment="Production",
                cancel_at_period_end=cancel_at_period_end,
                provider_event_created_at=BASE_TIME,
                now=BASE_TIME,
            )
        row = conn.execute(
            "SELECT cancel_at_period_end, current_period_end "
            "FROM billing_subscriptions WHERE external_subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        return int(row[0]), str(row[1])

    cancellation_then_active = final_state(True)
    active_then_cancellation = final_state(False)

    assert cancellation_then_active == active_then_cancellation
    assert cancellation_then_active == (
        1,
        (BASE_TIME + timedelta(days=20)).isoformat(),
    )


def test_stripe_signature_checkout_and_portal_use_server_configuration(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    account = {"id": account_id, "email": "billing@example.com"}
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "Sandbox")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_example")
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("BILLING_WEB_ORIGIN", "https://reelai.example")
    calls: dict[str, dict] = {}

    class Customer:
        @staticmethod
        def create(**kwargs):
            calls["customer"] = kwargs
            return {"id": "cus_1"}

    class CheckoutSession:
        @staticmethod
        def list(**_kwargs):
            return {"data": []}

        @staticmethod
        def create(**kwargs):
            calls["checkout"] = kwargs
            return {"url": "https://checkout.stripe.example/session"}

    class Subscription:
        @staticmethod
        def list(**_kwargs):
            return {"data": []}

    class PortalSession:
        @staticmethod
        def create(**kwargs):
            calls["portal"] = kwargs
            return {"url": "https://billing.stripe.example/portal"}

    class Webhook:
        @staticmethod
        def construct_event(payload, signature, secret):
            calls["signature"] = {
                "payload": payload,
                "signature": signature,
                "secret": secret,
            }
            return {"id": "evt_verified"}

    fake_stripe = SimpleNamespace(
        Customer=Customer,
        Subscription=Subscription,
        checkout=SimpleNamespace(Session=CheckoutSession),
        billing_portal=SimpleNamespace(Session=PortalSession),
        Webhook=Webhook,
        api_key=None,
    )
    monkeypatch.setattr(billing_providers, "_stripe_module", lambda: fake_stripe)

    assert billing_providers.create_stripe_checkout(
        conn, account=account, plan_code="plus"
    ) == "https://checkout.stripe.example/session"
    assert calls["checkout"]["payment_method_types"] == ["card"]
    assert calls["checkout"]["line_items"] == [{"price": "price_plus", "quantity": 1}]
    assert calls["checkout"]["client_reference_id"] == account_id
    assert calls["checkout"]["subscription_data"]["metadata"]["account_id"] == account_id
    assert calls["checkout"]["success_url"] == (
        "https://reelai.example/?settings=plan&checkout=success"
    )
    assert calls["checkout"]["cancel_url"] == (
        "https://reelai.example/?settings=plan&checkout=cancelled"
    )
    assert calls["checkout"]["idempotency_key"].startswith(
        f"reelai-checkout-sandbox-{account_id}-"
    )
    assert billing_providers.create_stripe_portal(
        conn, account_id=account_id
    ) == "https://billing.stripe.example/portal"
    assert calls["portal"]["customer"] == "cus_1"
    assert calls["portal"]["return_url"] == "https://reelai.example/?settings=plan"
    assert billing_providers.construct_stripe_event(b"payload", "signature") == {
        "id": "evt_verified"
    }
    assert calls["signature"] == {
        "payload": b"payload",
        "signature": "signature",
        "secret": "whsec_example",
    }


def test_stripe_price_ids_must_be_present_and_distinct(monkeypatch) -> None:
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_same")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_same")

    with pytest.raises(RuntimeError, match="must be different"):
        billing.stripe_price_for_plan("plus")
    with pytest.raises(RuntimeError, match="must be different"):
        billing.plan_for_product("stripe", "price_same")


def test_existing_subscription_rejects_a_different_unconfigured_price(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(),
    )

    def event(event_id: str, created: int, price_id: str) -> dict:
        return {
            "id": event_id,
            "type": "customer.subscription.updated",
            "created": created,
            "livemode": True,
            "data": {"object": {
                "id": "sub_price_pinned",
                "customer": "cus_price_pinned",
                "metadata": {"account_id": account_id},
                "items": {"data": [{
                    "price": {"id": price_id},
                    "current_period_end": int(
                        (BASE_TIME + timedelta(days=30)).timestamp()
                    ),
                }]},
                "status": "active",
            }},
        }

    assert billing_providers.process_stripe_event(
        conn,
        event("evt_known_price", 100, "price_pro"),
    )
    with pytest.raises(
        billing_providers.BillingVerificationError,
        match="unconfigured Price ID",
    ):
        billing_providers.process_stripe_event(
            conn,
            event("evt_unknown_price", 200, "price_unconfigured"),
        )

    row = conn.execute(
        "SELECT plan_code, external_product_id FROM billing_subscriptions "
        "WHERE external_subscription_id = 'sub_price_pinned'"
    ).fetchone()
    assert tuple(row) == ("pro", "price_pro")
    assert conn.execute(
        "SELECT COUNT(*) FROM billing_provider_events "
        "WHERE external_event_id = 'evt_unknown_price'"
    ).fetchone()[0] == 0


def test_full_stripe_refund_revokes_entitlement(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    subscription = {
        "id": "sub_refund",
        "customer": "cus_refund",
        "metadata": {"account_id": account_id},
        "items": {
            "data": [
                {
                    "price": {"id": "price_pro"},
                    "current_period_end": int(
                        (BASE_TIME + timedelta(days=30)).timestamp()
                    ),
                }
            ]
        },
        "status": "active",
        "cancel_at_period_end": False,
    }
    canceled: list[str] = []

    class Subscription:
        @staticmethod
        def retrieve(_subscription_id):
            return subscription

        @staticmethod
        def cancel(subscription_id):
            canceled.append(subscription_id)
            return {**subscription, "status": "canceled"}

    class Invoice:
        @staticmethod
        def retrieve(_invoice_id):
            return {
                "parent": {
                    "type": "subscription_details",
                    "subscription_details": {"subscription": "sub_refund"},
                }
            }

    class InvoicePayment:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {
                "payment": {
                    "type": "payment_intent",
                    "payment_intent": "pi_refund",
                },
                "limit": 1,
            }
            return {"data": [{"invoice": "in_refund"}]}

    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(
            Subscription=Subscription,
            Invoice=Invoice,
            InvoicePayment=InvoicePayment,
        ),
    )
    billing_providers.process_stripe_event(
        conn,
        {
            "id": "evt_initial",
            "type": "customer.subscription.updated",
            "created": 100,
            "livemode": True,
            "data": {"object": subscription},
        },
    )
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["plan"] == "pro"
    billing_providers.process_stripe_event(
        conn,
        {
            "id": "evt_refund",
            "type": "charge.refunded",
            "created": 200,
            "livemode": True,
            "data": {
                "object": {
                    "id": "ch_refund",
                    "amount": 1999,
                    "amount_refunded": 1999,
                    "payment_intent": "pi_refund",
                }
            },
        },
    )
    status = billing.billing_status(conn, account_id, now=BASE_TIME)
    assert status["plan"] == "free"
    assert status["subscriptions"][0]["status"] == "refunded"
    assert canceled == ["sub_refund"]


def test_legacy_billing_environment_migrations_fail_closed() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute("DROP TABLE billing_provider_customers")
    conn.execute("DROP TABLE billing_subscriptions")
    conn.execute(
        """
        CREATE TABLE billing_provider_customers (
            account_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            external_customer_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(account_id, provider),
            UNIQUE(provider, external_customer_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE billing_subscriptions (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            external_subscription_id TEXT NOT NULL,
            external_product_id TEXT NOT NULL,
            plan_code TEXT NOT NULL,
            status TEXT NOT NULL,
            current_period_end TEXT,
            cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
            provider_event_created_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(provider, external_subscription_id)
        )
        """
    )
    conn.execute(
        "INSERT INTO billing_provider_customers VALUES (?, 'stripe', 'cus_old', ?, ?)",
        (account_id, BASE_TIME.isoformat(), BASE_TIME.isoformat()),
    )
    conn.execute(
        "INSERT INTO billing_subscriptions VALUES (?, ?, 'stripe', 'sub_old', "
        "'price_old', 'pro', 'active', ?, 0, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            account_id,
            (BASE_TIME + timedelta(days=30)).isoformat(),
            BASE_TIME.isoformat(),
            BASE_TIME.isoformat(),
            BASE_TIME.isoformat(),
        ),
    )

    db._migrate_billing_provider_customer_environment_sqlite(conn)
    db._migrate_billing_subscription_environment_sqlite(conn)

    customer = conn.execute(
        "SELECT provider_environment FROM billing_provider_customers"
    ).fetchone()
    subscription = conn.execute(
        "SELECT provider_environment FROM billing_subscriptions"
    ).fetchone()
    assert customer[0] == "Unknown"
    assert subscription[0] == "Unknown"
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["plan"] == "free"


def test_partial_sqlite_customer_environment_migration_rebuilds_legacy_keys() -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute("DROP TABLE billing_provider_customers")
    conn.execute(
        """
        CREATE TABLE billing_provider_customers (
            account_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_environment TEXT NOT NULL DEFAULT 'Unknown',
            external_customer_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(account_id, provider),
            UNIQUE(provider, external_customer_id)
        )
        """
    )
    conn.execute(
        "INSERT INTO billing_provider_customers VALUES (?, 'stripe', 'Sandbox', "
        "'cus_sandbox', ?, ?)",
        (account_id, BASE_TIME.isoformat(), BASE_TIME.isoformat()),
    )

    db._migrate_billing_provider_customer_environment_sqlite(conn)

    primary_key = tuple(
        str(row[1])
        for row in sorted(
            conn.execute(
                "PRAGMA table_info(billing_provider_customers)"
            ).fetchall(),
            key=lambda row: int(row[5] or 0),
        )
        if int(row[5] or 0) > 0
    )
    assert primary_key == ("account_id", "provider", "provider_environment")
    assert conn.execute(
        "SELECT provider_environment FROM billing_provider_customers"
    ).fetchone()[0] == "Sandbox"
    conn.execute(
        "INSERT INTO billing_provider_customers VALUES (?, 'stripe', 'Production', "
        "'cus_production', ?, ?)",
        (account_id, BASE_TIME.isoformat(), BASE_TIME.isoformat()),
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM billing_provider_customers WHERE account_id = ?",
        (account_id,),
    ).fetchone()[0] == 2


def test_invalid_stripe_signature_is_a_billing_verification_error(monkeypatch) -> None:
    class SignatureVerificationError(Exception):
        pass

    class Webhook:
        @staticmethod
        def construct_event(_payload, _signature, _secret):
            raise SignatureVerificationError("bad signature")

    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_example")
    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(
            Webhook=Webhook,
            error=SimpleNamespace(
                SignatureVerificationError=SignatureVerificationError
            ),
        ),
    )
    with pytest.raises(billing_providers.BillingVerificationError):
        billing_providers.construct_stripe_event(b"payload", "invalid")


def test_incomplete_expired_stripe_subscription_allows_checkout_retry(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    _subscribe(
        conn,
        account_id,
        provider="stripe",
        plan="plus",
        status="incomplete_expired",
        provider_environment="Sandbox",
    )
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "Sandbox")
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("BILLING_WEB_ORIGIN", "https://reelai.example")

    class Customer:
        @staticmethod
        def create(**_kwargs):
            return {"id": "cus_retry"}

    class CheckoutSession:
        @staticmethod
        def list(**_kwargs):
            return {"data": []}

        @staticmethod
        def create(**_kwargs):
            return {"url": "https://checkout.stripe.example/retry"}

    class Subscription:
        @staticmethod
        def list(**_kwargs):
            return {"data": []}

    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(
            Customer=Customer,
            Subscription=Subscription,
            checkout=SimpleNamespace(Session=CheckoutSession),
        ),
    )
    assert billing_providers.create_stripe_checkout(
        conn,
        account={"id": account_id, "email": "retry@example.com"},
        plan_code="plus",
    ) == "https://checkout.stripe.example/retry"


def test_pending_checkout_reuses_same_plan_and_replaces_cross_plan(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "Sandbox")
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setenv("BILLING_WEB_ORIGIN", "https://reelai.example")
    open_sessions: list[dict] = []
    expired: list[str] = []
    created: list[str] = []

    class Customer:
        @staticmethod
        def create(**_kwargs):
            return {"id": "cus_pending"}

    class Subscription:
        @staticmethod
        def list(**_kwargs):
            return {"data": []}

    class CheckoutSession:
        @staticmethod
        def list(**_kwargs):
            return {"data": list(open_sessions)}

        @staticmethod
        def expire(session_id):
            expired.append(session_id)
            open_sessions[:] = [
                session for session in open_sessions if session["id"] != session_id
            ]

        @staticmethod
        def create(**kwargs):
            plan = str(kwargs["metadata"]["plan"])
            session = {
                "id": f"cs_{plan}",
                "mode": "subscription",
                "metadata": {"plan": plan},
                "url": f"https://checkout.stripe.example/{plan}",
            }
            created.append(plan)
            open_sessions.append(session)
            return session

    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(
            Customer=Customer,
            Subscription=Subscription,
            checkout=SimpleNamespace(Session=CheckoutSession),
        ),
    )
    account = {"id": account_id, "email": "pending@example.com"}
    plus_url = billing_providers.create_stripe_checkout(
        conn,
        account=account,
        plan_code="plus",
    )
    assert billing_providers.create_stripe_checkout(
        conn,
        account=account,
        plan_code="plus",
    ) == plus_url
    pro_url = billing_providers.create_stripe_checkout(
        conn,
        account=account,
        plan_code="pro",
    )
    assert pro_url == "https://checkout.stripe.example/pro"
    assert billing_providers.create_stripe_checkout(
        conn,
        account=account,
        plan_code="pro",
    ) == pro_url
    assert created == ["plus", "pro"]
    assert expired == ["cs_plus"]


def test_webhook_lagged_deletion_expires_checkout_and_cancels_remote_subscription(
    monkeypatch,
) -> None:
    conn = _conn()
    account_id = _account(conn)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    assert not billing_providers.cancel_stripe_for_account(conn, account_id)
    billing.upsert_provider_customer(
        conn,
        account_id=account_id,
        provider="stripe",
        provider_environment="Sandbox",
        external_customer_id="cus_lagged",
    )
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "Sandbox")
    expired: list[str] = []
    canceled: list[str] = []

    class CheckoutSession:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {
                "customer": "cus_lagged",
                "status": "open",
                "limit": 100,
            }
            return {"data": [{"id": "cs_open"}]}

        @staticmethod
        def expire(session_id):
            expired.append(session_id)

    class Subscription:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {
                "customer": "cus_lagged",
                "status": "all",
                "limit": 100,
            }
            return {
                "data": [
                    {"id": "sub_active", "status": "active"},
                    {"id": "sub_done", "status": "incomplete_expired"},
                ]
            }

        @staticmethod
        def cancel(subscription_id):
            canceled.append(subscription_id)

    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(
            checkout=SimpleNamespace(Session=CheckoutSession),
            Subscription=Subscription,
        ),
    )
    assert billing_providers.cancel_stripe_for_account(conn, account_id)
    assert expired == ["cs_open"]
    assert canceled == ["sub_active"]


def test_checkout_completion_racing_deleted_account_is_canceled(monkeypatch) -> None:
    conn = _conn()
    deleted_account_id = _account(conn)
    conn.execute("DELETE FROM community_accounts WHERE id = ?", (deleted_account_id,))
    canceled: list[str] = []
    subscription = {
        "id": "sub_orphan",
        "customer": "cus_orphan",
        "metadata": {"account_id": deleted_account_id},
        "status": "active",
    }

    class Subscription:
        @staticmethod
        def retrieve(_subscription_id):
            return subscription

        @staticmethod
        def cancel(subscription_id):
            canceled.append(subscription_id)

    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: SimpleNamespace(Subscription=Subscription),
    )
    assert billing_providers.process_stripe_event(
        conn,
        {
            "id": "evt_orphan_checkout",
            "type": "checkout.session.completed",
            "created": 100,
            "livemode": True,
            "data": {
                "object": {
                    "mode": "subscription",
                    "subscription": "sub_orphan",
                    "client_reference_id": deleted_account_id,
                }
            },
        },
    )
    assert canceled == ["sub_orphan"]




@pytest.mark.parametrize(("plan", "limit"), [("free", 5), ("plus", 15), ("pro", 50)])
def test_daily_boundaries(plan: str, limit: int) -> None:
    conn = _conn()
    account_id = _account(conn)
    if plan != "free":
        _subscribe(conn, account_id, provider="stripe", plan=plan)

    for index in range(limit):
        billing.reserve_search(
            conn,
            account_id=account_id,
            operation_key=f"material:{index}",
            surface="test",
            now=BASE_TIME,
            enforce=True,
        )
    with pytest.raises(billing.DailySearchLimitReached) as captured:
        billing.reserve_search(
            conn,
            account_id=account_id,
            operation_key="material:overflow",
            surface="test",
            now=BASE_TIME,
            enforce=True,
        )

    assert captured.value.plan == plan
    assert captured.value.limit == limit
    assert captured.value.used == limit
    assert captured.value.retry_after_seconds == 12 * 60 * 60
    assert captured.value.detail()["code"] == "daily_search_limit_reached"


def test_usage_resets_at_utc_midnight() -> None:
    conn = _conn()
    account_id = _account(conn)
    _subscribe(conn, account_id, provider="stripe", plan="pro")
    billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:today",
        surface="test",
        now=BASE_TIME,
        enforce=True,
    )

    today = billing.billing_status(conn, account_id, now=BASE_TIME)
    tomorrow = billing.billing_status(conn, account_id, now=BASE_TIME + timedelta(days=1))
    assert today["plan"] == "pro"
    assert today["used_searches"] == 1
    assert tomorrow["used_searches"] == 0
    assert tomorrow["remaining_searches"] == 50


def test_provider_environments_isolate_entitlements_and_customer_ids(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    _subscribe(
        conn,
        account_id,
        provider="stripe",
        plan="plus",
        provider_environment="Production",
    )
    _subscribe(
        conn,
        account_id,
        provider="stripe",
        plan="pro",
        provider_environment="Sandbox",
    )
    billing.upsert_provider_customer(
        conn,
        account_id=account_id,
        provider="stripe",
        provider_environment="Production",
        external_customer_id="cus_live",
    )
    billing.upsert_provider_customer(
        conn,
        account_id=account_id,
        provider="stripe",
        provider_environment="Sandbox",
        external_customer_id="cus_test",
    )

    assert billing.billing_status(conn, account_id, now=BASE_TIME)["plan"] == "plus"
    assert billing.provider_customer_id(conn, account_id, "stripe") == "cus_live"
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "Sandbox")
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["plan"] == "pro"
    assert billing.provider_customer_id(conn, account_id, "stripe") == "cus_test"
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "not-an-environment")
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["plan"] == "plus"


def test_stripe_livemode_environment_is_persisted(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setattr(billing_providers, "_stripe_module", lambda: SimpleNamespace())
    billing_providers.process_stripe_event(
        conn,
        {
            "id": "evt_sandbox",
            "type": "customer.subscription.updated",
            "created": 100,
            "livemode": False,
            "data": {
                "object": {
                    "id": "sub_sandbox",
                    "customer": "cus_sandbox",
                    "metadata": {"account_id": account_id},
                    "items": {
                        "data": [{
                            "price": {"id": "price_pro"},
                            "current_period_end": int((BASE_TIME + timedelta(days=30)).timestamp()),
                        }]
                    },
                    "status": "active",
                }
            },
        },
    )
    row = conn.execute(
        "SELECT provider_environment FROM billing_subscriptions "
        "WHERE external_subscription_id = 'sub_sandbox'"
    ).fetchone()
    assert row[0] == "Sandbox"
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["plan"] == "free"
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "Sandbox")
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["plan"] == "pro"


def test_equal_timestamp_terminal_subscription_state_cannot_be_revived() -> None:
    conn = _conn()
    account_id = _account(conn)
    _subscribe(
        conn,
        account_id,
        provider="stripe",
        plan="pro",
        event_time=BASE_TIME,
        status="refunded",
    )
    row = billing.upsert_subscription(
        conn,
        account_id=account_id,
        provider="stripe",
        provider_environment="Production",
        external_subscription_id=f"stripe-pro-{account_id}",
        external_product_id="stripe-pro",
        plan_code="pro",
        status="active",
        current_period_end=BASE_TIME + timedelta(days=30),
        provider_event_created_at=BASE_TIME,
        now=BASE_TIME,
    )
    assert row["status"] == "refunded"


def test_stripe_subscription_owner_and_environment_are_pinned(monkeypatch) -> None:
    conn = _conn()
    first_account = _account(conn)
    second_account = _account(conn)
    monkeypatch.setenv("STRIPE_PLUS_PRICE_ID", "price_plus")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_pro")
    monkeypatch.setattr(billing_providers, "_stripe_module", lambda: SimpleNamespace())

    def event(event_id: str, account_id: str, livemode: bool) -> dict:
        return {
            "id": event_id,
            "type": "customer.subscription.updated",
            "created": 100 if event_id == "evt_owner" else 200,
            "livemode": livemode,
            "data": {"object": {
                "id": "sub_pinned",
                "customer": "cus_pinned",
                "metadata": {"account_id": account_id},
                "items": {"data": [{
                    "price": {"id": "price_pro"},
                    "current_period_end": int((BASE_TIME + timedelta(days=30)).timestamp()),
                }]},
                "status": "active",
            }},
        }

    assert billing_providers.process_stripe_event(
        conn, event("evt_owner", first_account, True)
    )
    with pytest.raises(
        billing_providers.BillingVerificationError,
        match="another ReelAI account",
    ):
        billing_providers.process_stripe_event(
            conn, event("evt_steal", second_account, True)
        )
    with pytest.raises(
        billing_providers.BillingVerificationError,
        match="billing environment",
    ):
        billing_providers.process_stripe_event(
            conn, event("evt_environment", first_account, False)
        )
    row = conn.execute(
        "SELECT account_id, provider_environment FROM billing_subscriptions "
        "WHERE external_subscription_id = 'sub_pinned'"
    ).fetchone()
    assert tuple(row) == (first_account, "Production")
    assert conn.execute(
        "SELECT COUNT(*) FROM billing_provider_events "
        "WHERE external_event_id IN ('evt_steal', 'evt_environment')"
    ).fetchone()[0] == 0


def test_provider_configuration_and_checkout_origin_errors_are_normalized(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    billing.upsert_provider_customer(
        conn,
        account_id=account_id,
        provider="stripe",
        provider_environment="Sandbox",
        external_customer_id="cus_config",
    )
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("BILLING_ENTITLEMENT_ENVIRONMENT", "Sandbox")
    monkeypatch.setenv("BILLING_WEB_ORIGIN", "https://reelai.example")
    monkeypatch.delenv("STRIPE_PLUS_PRICE_ID", raising=False)
    monkeypatch.setattr(billing_providers, "_stripe_module", lambda: SimpleNamespace())
    with pytest.raises(
        billing_providers.BillingConfigurationError,
        match="STRIPE_PLUS_PRICE_ID",
    ):
        billing_providers.create_stripe_checkout(
            conn,
            account={"id": account_id, "email": "config@example.com"},
            plan_code="plus",
        )

    for invalid in (
        "http://studyreels.app",
        "https://user:pass@studyreels.app",
        "https://studyreels.app/account",
        "https://studyreels.app?next=bad",
        "https://studyreels.app:not-a-port",
        "https://:443",
    ):
        monkeypatch.setenv("BILLING_WEB_ORIGIN", invalid)
        with pytest.raises(billing_providers.BillingConfigurationError):
            billing_providers._web_origin()
    monkeypatch.setenv("BILLING_WEB_ORIGIN", "http://localhost:3000")
    assert billing_providers._web_origin() == "http://localhost:3000"
    monkeypatch.setenv("BILLING_WEB_ORIGIN", "https://studyreels.app/")
    assert billing_providers._web_origin() == "https://studyreels.app"


def test_stripe_only_http_contract_exposes_plans_and_requires_verified_account(
    monkeypatch,
    tmp_path,
) -> None:
    from fastapi.testclient import TestClient
    from backend.app.config import get_settings

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    from backend.app.main import app

    db._db_ready = False
    client = TestClient(app)

    plans = client.get("/api/billing/plans")
    assert plans.status_code == 200
    assert [plan["code"] for plan in plans.json()["plans"]] == [
        "free",
        "plus",
        "pro",
    ]
    status = client.get("/api/billing/status")
    assert status.status_code == 401
    assert status.json()["detail"]["code"] == "verified_account_required"
    assert client.post("/api/billing/apple/sync", json={}).status_code == 404
    assert client.post("/api/billing/apple/notifications", json={}).status_code == 404


def test_generation_active_completed_and_continuation_paths_do_not_create_search_charges(
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient
    from backend.app import main

    conn = _conn()
    account_id, material_id, headers = _authenticated_material(conn)
    monkeypatch.setenv("BILLING_ENFORCEMENT_ENABLED", "1")
    monkeypatch.setattr(
        main,
        "get_conn",
        lambda transactional=False: _fixed_connection(
            conn, transactional=transactional
        ),
    )
    monkeypatch.setattr(main, "_wake_generation_worker", lambda: None)
    client = TestClient(main.app)
    payload = {"material_id": material_id, "num_reels": 3}

    first = client.post("/api/reels/generate", json=payload, headers=headers)
    assert first.status_code == 202
    active_replay = client.post("/api/reels/generate", json=payload, headers=headers)
    assert active_replay.status_code == 202
    assert active_replay.json()["job_id"] == first.json()["job_id"]
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 0
    assert conn.execute("SELECT COUNT(*) FROM search_quota_reservations").fetchone()[0] == 0

    job_id = first.json()["job_id"]
    generation_jobs.append_event(
        conn,
        job_id=job_id,
        event_type="final",
        payload={"reels": [{"reel_id": "cached"}], "batch_size": 1},
        now=BASE_TIME,
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=job_id,
        status="completed",
        result_generation_id="generation-cached",
        now=BASE_TIME,
    )
    completed_replay = client.post(
        "/api/reels/generate", json=payload, headers=headers
    )
    assert completed_replay.status_code == 200
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 0

    continuation = client.post(
        "/api/reels/generate",
        json={**payload, "continuation_token": job_id},
        headers=headers,
    )
    assert continuation.status_code == 202
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 0
    assert conn.execute("SELECT COUNT(*) FROM search_quota_reservations").fetchone()[0] == 0


def test_feed_autofill_does_not_create_a_search_charge_at_the_daily_limit(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from backend.app import main

    conn = _conn()
    account_id, material_id, headers = _authenticated_material(conn)
    for index in range(5):
        reservation = billing.reserve_search(
            conn,
            account_id=account_id,
            operation_key=f"existing:{index}",
            surface="test",
            now=BASE_TIME,
            enforce=True,
        )
        billing.settle_reservation(
            conn,
            reservation_id=str(reservation["id"]),
            usable_result=True,
            now=BASE_TIME,
        )
    monkeypatch.setenv("BILLING_ENFORCEMENT_ENABLED", "1")
    monkeypatch.setattr(
        main,
        "get_conn",
        lambda transactional=False: _fixed_connection(
            conn, transactional=transactional
        ),
    )
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_wake_generation_worker", lambda: None)
    client = TestClient(main.app)
    response = client.get(
        "/api/feed",
        params={"material_id": material_id, "autofill": "true"},
        headers=headers,
    )
    assert response.status_code == 200, response.json()
    assert response.json()["generation_job_status"] == "queued"
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 5
    assert conn.execute(
        "SELECT COUNT(*) FROM search_quota_reservations WHERE operation_key = ?",
        (f"material:{material_id}",),
    ).fetchone()[0] == 0


def test_duration_cache_is_anonymous_but_a_cache_miss_requires_verified_account(
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient
    from backend.app import main

    conn = _conn()
    _account_id, _material_id, authenticated_headers = _authenticated_material(conn)
    anonymous_headers = {
        "x-studyreels-owner-key": authenticated_headers["x-studyreels-owner-key"]
    }
    monkeypatch.setattr(
        main,
        "get_conn",
        lambda transactional=False: _fixed_connection(
            conn, transactional=transactional
        ),
    )
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)

    class CachedTranscriptStore:
        @staticmethod
        def get_transcript(**kwargs):
            assert kwargs["provider"] == "supadata"
            return SimpleNamespace(duration_sec=91.5)

    monkeypatch.setattr(main, "DatabaseProviderCache", CachedTranscriptStore)
    monkeypatch.setattr(
        main,
        "_resolve_community_reel_duration_sec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached duration called the provider")
        ),
    )
    client = TestClient(main.app)
    params = {"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}

    cached = client.get(
        "/api/community/reels/duration",
        params=params,
        headers=anonymous_headers,
    )
    assert cached.status_code == 200
    assert cached.json() == {"duration_sec": 91.5}

    class EmptyTranscriptStore:
        @staticmethod
        def get_transcript(**_kwargs):
            return None

    provider_calls: list[str] = []
    monkeypatch.setattr(main, "DatabaseProviderCache", EmptyTranscriptStore)
    monkeypatch.setattr(
        main,
        "_resolve_community_reel_duration_sec",
        lambda source_url: provider_calls.append(source_url) or 92.0,
    )
    anonymous_miss = client.get(
        "/api/community/reels/duration",
        params=params,
        headers=anonymous_headers,
    )
    assert anonymous_miss.status_code == 401
    assert anonymous_miss.json()["detail"]["code"] == "verified_account_required"
    assert provider_calls == []

    verified_miss = client.get(
        "/api/community/reels/duration",
        params=params,
        headers=authenticated_headers,
    )
    assert verified_miss.status_code == 200
    assert verified_miss.json() == {"duration_sec": 92.0}
    assert provider_calls == [params["source_url"]]


def test_ingest_search_pagination_is_idempotent_and_uncharged_at_the_limit(
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient
    from backend.app import main

    conn = _conn()
    account_id, _material_id, headers = _authenticated_material(conn)
    for index in range(5):
        reservation = billing.reserve_search(
            conn,
            account_id=account_id,
            operation_key=f"existing:{index}",
            surface="test",
            now=BASE_TIME,
            enforce=True,
        )
        billing.settle_reservation(
            conn,
            reservation_id=str(reservation["id"]),
            usable_result=True,
            now=BASE_TIME,
        )
    monkeypatch.setenv("BILLING_ENFORCEMENT_ENABLED", "1")
    monkeypatch.setattr(main, "SERVERLESS_MODE", False)
    monkeypatch.setattr(
        main,
        "get_conn",
        lambda transactional=False: _fixed_connection(
            conn, transactional=transactional
        ),
    )
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    provider_calls: list[str] = []

    def ingest_search(**kwargs):
        provider_calls.append(str(kwargs["query"]))
        return main.IngestSearchResult(
            query=str(kwargs["query"]),
            material_id="ingest-scratch",
            platforms=["yt"],
            total_resolved=0,
            succeeded=0,
            failed=0,
            items=[],
            terms_notice="test",
            trace_id="trace-pagination",
        )

    monkeypatch.setattr(main.ingestion_pipeline, "ingest_search", ingest_search)
    client = TestClient(main.app)
    request_headers = {**headers, "Idempotency-Key": "pagination-operation"}
    body = {
        "query": "cell biology",
        "platforms": ["yt"],
        "exclude_video_ids": ["dQw4w9WgXcQ"],
    }

    first = client.post(
        "/api/ingest/search", json=body, headers=request_headers
    )
    replay = client.post(
        "/api/ingest/search", json=body, headers=request_headers
    )
    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert provider_calls == ["cell biology"]
    assert billing.billing_status(conn, account_id, now=BASE_TIME)["used_searches"] == 5
    assert conn.execute("SELECT COUNT(*) FROM search_quota_reservations").fetchone()[0] == 5


def test_ingest_idempotency_replays_when_quota_enforcement_is_disabled(
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient
    from backend.app import main

    conn = _conn()
    _account_id, _material_id, headers = _authenticated_material(conn)
    monkeypatch.setenv("BILLING_ENFORCEMENT_ENABLED", "0")
    monkeypatch.setattr(main, "SERVERLESS_MODE", False)
    monkeypatch.setattr(
        main,
        "get_conn",
        lambda transactional=False: _fixed_connection(
            conn, transactional=transactional
        ),
    )
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    provider_calls: list[str] = []

    def ingest_search(**kwargs):
        provider_calls.append(str(kwargs["query"]))
        return main.IngestSearchResult(
            query=str(kwargs["query"]),
            material_id="ingest-scratch",
            platforms=["yt"],
            total_resolved=0,
            succeeded=0,
            failed=0,
            items=[],
            terms_notice="test",
            trace_id="trace-disabled-enforcement",
        )

    monkeypatch.setattr(main.ingestion_pipeline, "ingest_search", ingest_search)
    client = TestClient(main.app)
    request_headers = {**headers, "Idempotency-Key": "disabled-enforcement-operation"}
    body = {"query": "cell biology", "platforms": ["yt"]}

    first = client.post(
        "/api/ingest/search", json=body, headers=request_headers
    )
    replay = client.post(
        "/api/ingest/search", json=body, headers=request_headers
    )
    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert provider_calls == ["cell biology"]
    assert conn.execute("SELECT COUNT(*) FROM daily_search_usage").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM search_quota_reservations").fetchone()[0] == 0


@pytest.mark.parametrize("enforcement", ["0", "1"])
def test_availability_probes_are_always_cache_only(monkeypatch, enforcement: str) -> None:
    from fastapi.testclient import TestClient
    from backend.app import main

    conn = _conn()
    _account_id, material_id, headers = _authenticated_material(conn)
    monkeypatch.setenv("BILLING_ENFORCEMENT_ENABLED", enforcement)
    monkeypatch.setattr(
        main,
        "get_conn",
        lambda transactional=False: _fixed_connection(
            conn, transactional=transactional
        ),
    )
    monkeypatch.setattr(
        main,
        "supadata_search_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("availability probe called the provider")
        ),
    )
    client = TestClient(main.app)
    response = client.post(
        "/api/reels/can-generate",
        json={"material_id": material_id},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["availability"] == "unknown"
    any_response = client.post(
        "/api/reels/can-generate-any",
        json={"material_ids": [material_id]},
        headers=headers,
    )
    assert any_response.status_code == 200
    assert any_response.json()["availability"] == "unknown"


def test_provider_account_gate_remains_enabled_when_quota_enforcement_is_off(
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient
    from backend.app import main

    conn = _conn()
    _account_id, material_id, authenticated_headers = _authenticated_material(conn)
    anonymous_headers = {
        "x-studyreels-owner-key": authenticated_headers["x-studyreels-owner-key"]
    }
    monkeypatch.setenv("BILLING_ENFORCEMENT_ENABLED", "0")
    monkeypatch.setattr(main, "SERVERLESS_MODE", False)
    monkeypatch.setattr(
        main,
        "get_conn",
        lambda transactional=False: _fixed_connection(
            conn, transactional=transactional
        ),
    )
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_wake_generation_worker", lambda: None)
    monkeypatch.setattr(
        main.material_intelligence_service,
        "extract_concepts_and_objectives",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("material provider was called")
        ),
    )
    monkeypatch.setattr(
        main.ingestion_pipeline,
        "ingest_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ingest provider was called")
        ),
    )
    monkeypatch.setattr(
        main.material_intelligence_service,
        "chat_assistant",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("chat provider was called")
        ),
    )
    client = TestClient(main.app)

    responses = [
        client.post(
            "/api/material",
            data={"subject_tag": "Cell biology"},
            headers=anonymous_headers,
        ),
        client.post(
            "/api/reels/generate",
            json={"material_id": material_id},
            headers=anonymous_headers,
        ),
        client.post(
            "/api/ingest/search",
            json={"query": "Cell biology", "platforms": ["yt"]},
            headers=anonymous_headers,
        ),
        client.post(
            "/api/chat",
            json={"message": "Explain mitosis."},
            headers=anonymous_headers,
        ),
    ]
    for response in responses:
        assert response.status_code == 401, response.text
        assert response.json()["detail"]["code"] == "verified_account_required"
    assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 0

    allowed = client.post(
        "/api/reels/generate",
        json={"material_id": material_id},
        headers=authenticated_headers,
    )
    assert allowed.status_code == 202, allowed.text
    assert conn.execute("SELECT COUNT(*) FROM daily_search_usage").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM search_quota_reservations").fetchone()[0] == 0


def test_checkout_fails_closed_if_account_disappeared_before_lock(monkeypatch) -> None:
    conn = _conn()
    account_id = _account(conn)
    conn.execute("DELETE FROM community_accounts WHERE id = ?", (account_id,))
    monkeypatch.setattr(
        billing_providers,
        "_stripe_module",
        lambda: (_ for _ in ()).throw(AssertionError("Stripe was called")),
    )

    with pytest.raises(billing_providers.BillingAccountNotFoundError):
        billing_providers.create_stripe_checkout(
            conn,
            account={"id": account_id, "email": "deleted@example.com"},
            plan_code="plus",
        )


def test_highest_simultaneous_active_stripe_tier_wins() -> None:
    conn = _conn()
    account_id = _account(conn)
    _subscribe(conn, account_id, provider="stripe", plan="plus")
    _subscribe(conn, account_id, provider="stripe", plan="pro")
    status = billing.billing_status(conn, account_id, now=BASE_TIME)
    assert status["plan"] == "pro"
    assert status["daily_limit"] == 50
    reservation = billing.reserve_search(
        conn,
        account_id=account_id,
        operation_key="material:cost-snapshot",
        surface="test",
        now=BASE_TIME,
        enforce=True,
    )
    assert reservation["plan_code"] == "pro"


def test_active_stripe_subscription_without_expiry_does_not_grant_access() -> None:
    conn = _conn()
    account_id = _account(conn)
    billing.upsert_subscription(
        conn,
        account_id=account_id,
        provider="stripe",
        external_subscription_id=f"stripe-plus-{account_id}",
        external_product_id="stripe-plus",
        plan_code="plus",
        status="active",
        current_period_end=None,
        provider_environment="Production",
        provider_event_created_at=BASE_TIME,
        now=BASE_TIME,
    )

    status = billing.billing_status(conn, account_id, now=BASE_TIME)

    assert status["plan"] == "free"
    assert status["daily_limit"] == 5
