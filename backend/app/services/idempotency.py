"""Exact transport-idempotency records for expensive API submissions."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..db import (
    DatabaseIntegrityError,
    dumps_json,
    execute_modify,
    fetch_one,
    insert,
    loads_json,
    now_iso,
)


class IdempotencyConflictError(ValueError):
    """One client key was reused with a different request body."""


@dataclass(frozen=True)
class IdempotencyReservation:
    resource_id: str
    owner: bool
    status: str
    response: dict[str, Any] | None
    attempt_token: str | None = None
    reclaimed: bool = False


def normalize_idempotency_key(raw_value: object) -> str | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    if len(value) > 200 or any(ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise ValueError("Idempotency-Key must contain 1-200 visible ASCII characters.")
    return value


def idempotency_key_hash(raw_value: str) -> str:
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def request_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reserve_idempotency_key(
    conn: Any,
    *,
    scope: str,
    learner_id: str,
    raw_key: str,
    fingerprint: str,
    resource_id: str,
    stale_after_seconds: int | None = None,
) -> IdempotencyReservation:
    """Reserve a key once, or return the exact existing reservation."""
    clean_scope = str(scope or "").strip()
    clean_learner = str(learner_id or "").strip()
    clean_fingerprint = str(fingerprint or "").strip()
    clean_resource_id = str(resource_id or "").strip()
    if not all((clean_scope, clean_learner, clean_fingerprint, clean_resource_id)):
        raise ValueError("scope, learner_id, fingerprint, and resource_id are required")
    key_hash = idempotency_key_hash(raw_key)
    created_at = now_iso()
    attempt_token = secrets.token_hex(16)
    try:
        insert(
            conn,
            "api_idempotency_records",
            {
                "scope": clean_scope,
                "learner_id": clean_learner,
                "key_hash": key_hash,
                "request_fingerprint": clean_fingerprint,
                "status": "in_progress",
                "resource_id": clean_resource_id,
                "attempt_token": attempt_token,
                "response_json": None,
                "created_at": created_at,
                "updated_at": created_at,
            },
        )
        return IdempotencyReservation(
            resource_id=clean_resource_id,
            owner=True,
            status="in_progress",
            response=None,
            attempt_token=attempt_token,
        )
    except DatabaseIntegrityError:
        row = fetch_one(
            conn,
            "SELECT request_fingerprint, status, resource_id, attempt_token, "
            "response_json, updated_at "
            "FROM api_idempotency_records "
            "WHERE scope = ? AND learner_id = ? AND key_hash = ?",
            (clean_scope, clean_learner, key_hash),
        )
        if not row:
            raise
        if str(row.get("request_fingerprint") or "") != clean_fingerprint:
            raise IdempotencyConflictError(
                "Idempotency-Key was already used with a different request."
            )
        status = str(row.get("status") or "in_progress")
        if status == "in_progress" and stale_after_seconds is not None:
            try:
                updated_at = datetime.fromisoformat(
                    str(row.get("updated_at") or "").replace("Z", "+00:00")
                )
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                age_seconds = (
                    datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)
                ).total_seconds()
            except (TypeError, ValueError):
                age_seconds = float("inf")
            if age_seconds >= max(1, int(stale_after_seconds)):
                reclaimed_at = now_iso()
                reclaimed_token = secrets.token_hex(16)
                if execute_modify(
                    conn,
                    "UPDATE api_idempotency_records "
                    "SET attempt_token = ?, updated_at = ? "
                    "WHERE scope = ? AND learner_id = ? AND key_hash = ? "
                    "AND resource_id = ? AND status = 'in_progress' "
                    "AND attempt_token = ? AND updated_at = ?",
                    (
                        reclaimed_token,
                        reclaimed_at,
                        clean_scope,
                        clean_learner,
                        key_hash,
                        str(row.get("resource_id") or ""),
                        str(row.get("attempt_token") or ""),
                        str(row.get("updated_at") or ""),
                    ),
                ):
                    return IdempotencyReservation(
                        resource_id=str(row.get("resource_id") or ""),
                        owner=True,
                        status="in_progress",
                        response=None,
                        attempt_token=reclaimed_token,
                        reclaimed=True,
                    )
        response = loads_json(str(row.get("response_json") or ""), default=None)
        return IdempotencyReservation(
            resource_id=str(row.get("resource_id") or ""),
            owner=False,
            status=status,
            response=response if isinstance(response, dict) else None,
        )


def complete_idempotency_key(
    conn: Any,
    *,
    scope: str,
    learner_id: str,
    raw_key: str,
    resource_id: str,
    attempt_token: str,
    response: dict[str, Any],
) -> bool:
    if not str(attempt_token or "").strip():
        return False
    return bool(
        execute_modify(
            conn,
            "UPDATE api_idempotency_records "
            "SET status = 'completed', response_json = ?, updated_at = ? "
            "WHERE scope = ? AND learner_id = ? AND key_hash = ? "
            "AND resource_id = ? AND attempt_token = ? "
            "AND status = 'in_progress'",
            (
                dumps_json(response),
                now_iso(),
                scope,
                learner_id,
                idempotency_key_hash(raw_key),
                resource_id,
                attempt_token,
            ),
        )
    )

def lock_idempotency_attempt(
    conn: Any,
    *,
    scope: str,
    learner_id: str,
    raw_key: str,
    resource_id: str,
    attempt_token: str,
) -> bool:
    """Fence and row-lock the current owner inside its publish transaction."""
    if not str(attempt_token or "").strip():
        return False
    return bool(
        execute_modify(
            conn,
            "UPDATE api_idempotency_records SET updated_at = ? "
            "WHERE scope = ? AND learner_id = ? AND key_hash = ? "
            "AND resource_id = ? AND attempt_token = ? "
            "AND status = 'in_progress'",
            (
                now_iso(),
                scope,
                learner_id,
                idempotency_key_hash(raw_key),
                resource_id,
                attempt_token,
            ),
        )
    )


def finalize_idempotency_resource(
    conn: Any,
    *,
    scope: str,
    resource_id: str,
) -> bool:
    """Mark a durable resource mapping final without duplicating its response body."""
    return bool(
        execute_modify(
            conn,
            "UPDATE api_idempotency_records "
            "SET status = 'completed', updated_at = ? "
            "WHERE scope = ? AND resource_id = ?",
            (now_iso(), str(scope), str(resource_id)),
        )
    )


def release_idempotency_key(
    conn: Any,
    *,
    scope: str,
    learner_id: str,
    raw_key: str,
    resource_id: str,
    attempt_token: str,
) -> bool:
    """Release only the failed owner so a deliberate retry can run again."""
    if not str(attempt_token or "").strip():
        return False
    return bool(
        execute_modify(
            conn,
            "DELETE FROM api_idempotency_records "
            "WHERE scope = ? AND learner_id = ? AND key_hash = ? "
            "AND resource_id = ? AND attempt_token = ? "
            "AND status = 'in_progress'",
            (
                scope,
                learner_id,
                idempotency_key_hash(raw_key),
                resource_id,
                attempt_token,
            ),
        )
    )
