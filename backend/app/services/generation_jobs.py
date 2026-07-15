"""Database-backed generation job lifecycle and replayable event storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal

from ..db import (
    DatabaseIntegrityError,
    dumps_json,
    execute_modify,
    fetch_all,
    fetch_one,
    insert,
    loads_json,
)


JobStatus = Literal[
    "queued",
    "running",
    "completed",
    "partial",
    "exhausted",
    "failed",
    "cancelled",
]
EventType = Literal["candidate", "final", "terminal"]

ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"completed", "partial", "exhausted", "failed", "cancelled"})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
EVENT_TYPES = frozenset({"candidate", "final", "terminal"})

DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_HEARTBEAT_SECONDS = 15
DEFAULT_LEASE_SECONDS = 90
DEFAULT_DEADLINE_SECONDS = 60 * 60
DEFAULT_QUEUE_TTL_SECONDS = 8 * 60
# Request-key version doubles as a production inventory compatibility gate.
REQUEST_SCHEMA_VERSION = "quality_silence_v15"
GENERATION_SUBMIT_ADVISORY_LOCK_ID = 0x5354554459524545


class JobLeaseLostError(RuntimeError):
    """The worker no longer owns a live lease for the requested write."""


class GenerationQueueFullError(RuntimeError):
    """A distinct request would exceed the configured active-job ceiling."""

    def __init__(self, *, scope: Literal["global", "learner"], limit: int) -> None:
        self.scope = scope
        self.limit = int(limit)
        super().__init__(f"{scope} active generation limit reached ({limit})")


def _utc_now(value: datetime | str | None = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | str | None = None) -> str:
    return _utc_now(value).isoformat()


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


@contextmanager
def _atomic_write(conn: Any):
    """Make multi-statement repository writes atomic on autocommit connections.

    Callers that supplied an explicit transaction retain ownership of it. This
    keeps terminal state and its terminal event from being committed separately
    when the normal application connection is in autocommit mode.
    """
    is_sqlite = isinstance(conn, sqlite3.Connection)
    owns_sqlite_transaction = bool(
        is_sqlite and conn.isolation_level is None and not conn.in_transaction
    )
    owns_postgres_transaction = bool(
        not is_sqlite and getattr(conn, "autocommit", False)
    )
    if owns_sqlite_transaction:
        conn.execute("BEGIN IMMEDIATE")
    elif owns_postgres_transaction:
        conn.autocommit = False
    try:
        yield
    except Exception:
        if owns_sqlite_transaction or owns_postgres_transaction:
            conn.rollback()
        raise
    else:
        if owns_sqlite_transaction or owns_postgres_transaction:
            conn.commit()
    finally:
        if owns_postgres_transaction:
            conn.autocommit = True


def material_content_fingerprint(conn: Any, material_id: str, concept_id: str | None = None) -> str:
    """Hash the material and selected concept content without renaming concept IDs."""
    try:
        material = fetch_one(
            conn,
            "SELECT id, subject_tag, raw_text, source_type FROM materials WHERE id = ?",
            (material_id,),
        )
    except Exception as exc:
        if "column" not in str(exc).lower():
            raise
        material = fetch_one(conn, "SELECT * FROM materials WHERE id = ?", (material_id,))
    if not material:
        raise ValueError("material not found")
    params: tuple[Any, ...] = (material_id,)
    concept_where = "material_id = ?"
    if concept_id:
        concept_where += " AND id = ?"
        params = (material_id, concept_id)
    try:
        concepts = fetch_all(
            conn,
            "SELECT id, title, keywords_json, summary "
            f"FROM concepts WHERE {concept_where} ORDER BY id",
            params,
        )
    except Exception as exc:
        if "no such table" not in str(exc).lower() and "column" not in str(exc).lower():
            raise
        try:
            concepts = fetch_all(
                conn,
                f"SELECT * FROM concepts WHERE {concept_where} ORDER BY id",
                params,
            )
        except Exception as fallback_exc:
            if "no such table" not in str(fallback_exc).lower():
                raise
            concepts = []
    if concept_id and not concepts:
        raise ValueError("concept not found for material")
    normalized_concepts: list[dict[str, Any]] = []
    for concept in concepts:
        keywords = loads_json(str(concept.get("keywords_json") or "[]"), default=[])
        normalized_concepts.append(
            {
                "id": str(concept.get("id") or ""),
                "title": _normalize_text(concept.get("title")),
                "keywords": keywords if isinstance(keywords, list) else [],
                "summary": _normalize_text(concept.get("summary")),
            }
        )
    payload = {
        "material_id": str(material.get("id") or material_id),
        "subject_tag": _normalize_text(material.get("subject_tag")),
        "raw_text": _normalize_text(material.get("raw_text")),
        "source_type": _normalize_text(material.get("source_type")).lower(),
        "concept_id": concept_id or "",
        "concepts": normalized_concepts,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_request_key(
    *,
    material_id: str,
    concept_id: str | None,
    content_fingerprint: str,
    knowledge_level: str,
    generation_mode: Literal["slow", "fast"],
    creative_commons_only: bool,
    source_duration: str,
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
    language: str = "en",
    min_relevance: float | None = None,
    exclude_video_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> str:
    """Build the normalized request key; deprecated clip-duration fields are inert."""
    payload = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "material_id": str(material_id),
        "concept_id": str(concept_id or ""),
        "content_fingerprint": str(content_fingerprint),
        "knowledge_level": _normalize_text(knowledge_level).lower(),
        "generation_mode": generation_mode,
        "creative_commons_only": bool(creative_commons_only),
        "source_duration": _normalize_text(source_duration).lower() or "any",
        "language": _normalize_text(language).lower() or "en",
        "min_relevance": (
            None if min_relevance is None else round(float(min_relevance), 6)
        ),
        "exclude_video_ids": sorted(
            {
                _normalize_text(video_id).removeprefix("yt:")
                for video_id in (exclude_video_ids or [])
                if _normalize_text(video_id)
            }
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get_job(conn: Any, job_id: str) -> dict[str, Any] | None:
    return fetch_one(conn, "SELECT * FROM reel_generation_jobs WHERE id = ?", (job_id,))


def find_active_job(
    conn: Any,
    request_key: str,
    *,
    now: datetime | str | None = None,
) -> dict[str, Any] | None:
    active = fetch_one(
        conn,
        """
        SELECT * FROM reel_generation_jobs
        WHERE request_key = ? AND status IN ('queued', 'running')
        ORDER BY created_at, id
        LIMIT 1
        """,
        (request_key,),
    )
    if active and expire_stale_queued_job(conn, job_id=str(active["id"]), now=now):
        return fetch_one(
            conn,
            """
            SELECT * FROM reel_generation_jobs
            WHERE request_key = ? AND status IN ('queued', 'running')
            ORDER BY created_at, id
            LIMIT 1
            """,
            (request_key,),
        )
    return active


def find_completed_job(conn: Any, request_key: str) -> dict[str, Any] | None:
    return fetch_one(
        conn,
        """
        SELECT * FROM reel_generation_jobs
        WHERE request_key = ?
          AND status IN ('completed', 'partial', 'exhausted')
        ORDER BY completed_at DESC, created_at DESC, id DESC
        LIMIT 1
        """,
        (request_key,),
    )


def _acquire_generation_submit_lock(conn: Any) -> None:
    if isinstance(conn, sqlite3.Connection):
        return
    fetch_one(
        conn,
        "SELECT pg_advisory_xact_lock(?) AS acquired",
        (GENERATION_SUBMIT_ADVISORY_LOCK_ID,),
    )


def submit_or_get_active(
    conn: Any,
    *,
    material_id: str,
    concept_id: str | None,
    request_key: str,
    content_fingerprint: str,
    learner_id: str,
    request_params: dict[str, Any],
    source_generation_id: str | None = None,
    target_profile: str = "unified",
    now: datetime | str | None = None,
    deadline_seconds: int = DEFAULT_DEADLINE_SECONDS,
    job_id: str | None = None,
    max_global_active_jobs: int | None = None,
    max_active_jobs_per_learner: int | None = None,
    before_create: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Atomically create one queued job or return the active request-key winner."""
    if not str(material_id or "").strip():
        raise ValueError("material_id is required")
    if not str(request_key or "").strip():
        raise ValueError("request_key is required")
    if not str(content_fingerprint or "").strip():
        raise ValueError("content_fingerprint is required")
    started = _utc_now(now)
    created_at = started.isoformat()
    bounded_deadline = max(1, min(DEFAULT_DEADLINE_SECONDS, int(deadline_seconds)))
    deadline_at = (started + timedelta(seconds=bounded_deadline)).isoformat()
    requested_id = str(job_id or "").strip()
    active = find_active_job(conn, request_key, now=started)
    if active:
        return active, False
    _acquire_generation_submit_lock(conn)
    active = find_active_job(conn, request_key, now=started)
    if active:
        return active, False
    if max_global_active_jobs is not None or max_active_jobs_per_learner is not None:
        counts = fetch_one(
            conn,
            """
            SELECT COUNT(*) AS global_active,
                   COALESCE(SUM(CASE WHEN learner_id = ? THEN 1 ELSE 0 END), 0)
                       AS learner_active
            FROM reel_generation_jobs
            WHERE status IN ('queued', 'running')
            """,
            (str(learner_id or "legacy"),),
        ) or {}
        learner_active = int(counts.get("learner_active") or 0)
        global_active = int(counts.get("global_active") or 0)
        if (
            max_active_jobs_per_learner is not None
            and learner_active >= int(max_active_jobs_per_learner)
        ):
            raise GenerationQueueFullError(
                scope="learner",
                limit=int(max_active_jobs_per_learner),
            )
        if (
            max_global_active_jobs is not None
            and global_active >= int(max_global_active_jobs)
        ):
            raise GenerationQueueFullError(
                scope="global",
                limit=int(max_global_active_jobs),
            )
    if before_create is not None:
        before_create()
    for attempt in range(3):
        candidate_id = requested_id if attempt == 0 and requested_id else str(uuid.uuid4())
        row = {
            "id": candidate_id,
            "material_id": material_id,
            "concept_id": concept_id,
            "request_key": request_key,
            "content_fingerprint": content_fingerprint,
            "learner_id": str(learner_id or "legacy"),
            "source_generation_id": str(source_generation_id or ""),
            "result_generation_id": None,
            "target_profile": target_profile,
            "request_params_json": dumps_json(request_params),
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "lease_owner": None,
            "lease_expires_at": None,
            "heartbeat_at": None,
            "attempt_count": 0,
            "max_attempts": DEFAULT_MAX_ATTEMPTS,
            "deadline_at": deadline_at,
            "cancel_requested": 0,
            "cancel_requested_at": None,
            "model_used": None,
            "quality_degraded": 0,
            "usage_json": "{}",
            "terminal_error_code": None,
            "terminal_error_message": None,
            "terminal_error_json": None,
            "next_event_seq": 0,
            "created_at": created_at,
            "updated_at": created_at,
            "started_at": None,
            "completed_at": None,
            "error_text": None,
        }
        try:
            insert(conn, "reel_generation_jobs", row)
            return row, True
        except DatabaseIntegrityError:
            active = find_active_job(conn, request_key, now=started)
            if active:
                return active, False
    active = find_active_job(conn, request_key, now=started)
    if active:
        return active, False
    raise RuntimeError("could not submit or locate an active generation job")


def _terminal_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(row.get("status") or ""),
        "result_generation_id": str(row.get("result_generation_id") or "") or None,
        "model_used": str(row.get("model_used") or "") or None,
        "quality_degraded": bool(row.get("quality_degraded")),
        "usage": loads_json(str(row.get("usage_json") or "{}"), default={}),
        "error": {
            "code": str(row.get("terminal_error_code") or "") or None,
            "message": str(row.get("terminal_error_message") or "") or None,
            "detail": loads_json(str(row.get("terminal_error_json") or ""), default=None),
        },
    }


def append_event(
    conn: Any,
    *,
    job_id: str,
    event_type: EventType,
    payload: dict[str, Any],
    lease_owner: str | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported generation event type: {event_type}")
    if lease_owner is not None and not str(lease_owner).strip():
        raise ValueError("lease_owner cannot be blank")
    created_at = _iso(now)
    owner_clause = ""
    params: list[Any] = [created_at, job_id]
    if lease_owner is not None:
        owner_clause = (
            " AND status = 'running' AND lease_owner = ? AND cancel_requested = 0"
            " AND lease_expires_at > ?"
            " AND (deadline_at IS NULL OR deadline_at > ?)"
        )
        params.extend([lease_owner, created_at, created_at])
    with _atomic_write(conn):
        seq_row = fetch_one(
            conn,
            """
            UPDATE reel_generation_jobs
            SET next_event_seq = next_event_seq + 1,
                updated_at = ?
            WHERE id = ?
            """
            + owner_clause
            + """
            RETURNING next_event_seq
            """,
            params,
        )
        if not seq_row:
            if get_job(conn, job_id) is None:
                raise KeyError(f"generation job not found: {job_id}")
            raise JobLeaseLostError(f"generation job lease is no longer active: {job_id}")
        seq = int(seq_row.get("next_event_seq") or 0)
        insert(
            conn,
            "generation_job_events",
            {
                "job_id": job_id,
                "seq": seq,
                "event_type": event_type,
                "payload_json": dumps_json(payload),
                "created_at": created_at,
            },
        )
    return {
        "job_id": job_id,
        "seq": seq,
        "timestamp": created_at,
        "type": event_type,
        "payload": payload,
    }


def replay_events(
    conn: Any,
    *,
    job_id: str,
    after_seq: int = 0,
    limit: int = 500,
) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT job_id, seq, event_type, payload_json, created_at
        FROM generation_job_events
        WHERE job_id = ? AND seq > ?
        ORDER BY seq
        LIMIT ?
        """,
        (job_id, max(0, int(after_seq)), max(1, min(2000, int(limit)))),
    )
    return [
        {
            "job_id": str(row.get("job_id") or job_id),
            "seq": int(row.get("seq") or 0),
            "timestamp": str(row.get("created_at") or ""),
            "type": str(row.get("event_type") or ""),
            "payload": loads_json(str(row.get("payload_json") or "{}"), default={}),
        }
        for row in rows
    ]


def expire_stale_queued_job(
    conn: Any,
    *,
    job_id: str,
    now: datetime | str | None = None,
    queue_ttl_seconds: int = DEFAULT_QUEUE_TTL_SECONDS,
) -> bool:
    """Fail an unstarted job once after its bounded queue wait elapses."""
    current = get_job(conn, job_id)
    if not current or str(current.get("status") or "") != "queued":
        return False
    if int(current.get("attempt_count") or 0) != 0:
        return False
    now_dt = _utc_now(now)
    ttl_seconds = max(1, int(queue_ttl_seconds))
    cutoff = now_dt - timedelta(seconds=ttl_seconds)
    try:
        if _utc_now(current.get("created_at")) > cutoff:
            return False
    except (TypeError, ValueError):
        return False

    timestamp = now_dt.isoformat()
    cutoff_text = cutoff.isoformat()
    with _atomic_write(conn):
        updated = execute_modify(
            conn,
            """
            UPDATE reel_generation_jobs
            SET status = 'failed',
                phase = 'terminal',
                progress = 1.0,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                completed_at = ?,
                updated_at = ?,
                terminal_error_code = 'queue_timeout',
                terminal_error_message = 'Generation job waited too long to start.',
                terminal_error_json = NULL,
                error_text = 'queue_timeout'
            WHERE id = ?
              AND status = 'queued'
              AND attempt_count = 0
              AND created_at <= ?
            """,
            (timestamp, timestamp, job_id, cutoff_text),
        )
        if not updated:
            return False
        row = get_job(conn, job_id)
        if row:
            append_event(
                conn,
                job_id=job_id,
                event_type="terminal",
                payload=_terminal_payload(row),
                now=timestamp,
            )
    return True


def _fail_unclaimable_job(conn: Any, job_id: str, now_text: str) -> None:
    with _atomic_write(conn):
        updated = execute_modify(
            conn,
            """
        UPDATE reel_generation_jobs
        SET status = 'failed',
            phase = 'terminal',
            progress = 1.0,
            lease_owner = NULL,
            lease_expires_at = NULL,
            heartbeat_at = NULL,
            completed_at = ?,
            updated_at = ?,
            terminal_error_code = CASE
                WHEN deadline_at IS NOT NULL AND deadline_at <= ? THEN 'deadline_exceeded'
                ELSE 'attempts_exhausted'
            END,
            terminal_error_message = CASE
                WHEN deadline_at IS NOT NULL AND deadline_at <= ? THEN 'Generation job exceeded its deadline.'
                ELSE 'Generation job exhausted its lease attempts.'
            END,
            error_text = CASE
                WHEN deadline_at IS NOT NULL AND deadline_at <= ? THEN 'deadline_exceeded'
                ELSE 'attempts_exhausted'
            END
        WHERE id = ?
          AND status IN ('queued', 'running')
          AND (
              (deadline_at IS NOT NULL AND deadline_at <= ?)
              OR (
                  attempt_count >= max_attempts
                  AND (status = 'queued' OR lease_expires_at IS NULL OR lease_expires_at <= ?)
              )
          )
            """,
            (now_text, now_text, now_text, now_text, now_text, job_id, now_text, now_text),
        )
        if updated:
            row = get_job(conn, job_id)
            if row:
                append_event(
                    conn,
                    job_id=job_id,
                    event_type="terminal",
                    payload=_terminal_payload(row),
                    now=now_text,
                )


def lease_job(
    conn: Any,
    *,
    job_id: str,
    lease_owner: str,
    now: datetime | str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> dict[str, Any] | None:
    """Claim a queued job or reclaim an expired lease with one conditional update."""
    if not str(lease_owner or "").strip():
        raise ValueError("lease_owner is required")
    now_dt = _utc_now(now)
    now_text = now_dt.isoformat()
    if expire_stale_queued_job(conn, job_id=job_id, now=now_dt):
        return None
    current = get_job(conn, job_id)
    if not current:
        return None
    first_attempt = (
        str(current.get("status") or "") == "queued"
        and int(current.get("attempt_count") or 0) == 0
    )
    if first_attempt:
        # Queue time must not consume the execution window. Before the first
        # lease, deadline_at - created_at retains the requested window length.
        try:
            raw_submitted_at = current.get("created_at")
            raw_submitted_deadline = current.get("deadline_at")
            if not raw_submitted_at or not raw_submitted_deadline:
                raise ValueError("missing submitted execution window")
            submitted_at = _utc_now(raw_submitted_at)
            submitted_deadline = _utc_now(raw_submitted_deadline)
            execution_seconds = int((submitted_deadline - submitted_at).total_seconds())
        except (TypeError, ValueError):
            execution_seconds = DEFAULT_DEADLINE_SECONDS
        execution_seconds = max(1, min(DEFAULT_DEADLINE_SECONDS, execution_seconds))
        deadline_at = (now_dt + timedelta(seconds=execution_seconds)).isoformat()
    else:
        raw_deadline = current.get("deadline_at")
        deadline_at = _iso(raw_deadline) if raw_deadline else None
    lease_duration = max(1, min(DEFAULT_LEASE_SECONDS, int(lease_seconds)))
    lease_expires_dt = now_dt + timedelta(seconds=lease_duration)
    if deadline_at:
        lease_expires_dt = min(lease_expires_dt, _utc_now(deadline_at))
    lease_expires_at = lease_expires_dt.isoformat()
    queue_cutoff = (now_dt - timedelta(seconds=DEFAULT_QUEUE_TTL_SECONDS)).isoformat()
    claimed = execute_modify(
        conn,
        """
        UPDATE reel_generation_jobs
        SET status = 'running',
            phase = CASE WHEN phase = 'queued' THEN 'starting' ELSE phase END,
            lease_owner = ?,
            deadline_at = CASE
                WHEN status = 'queued' AND attempt_count = 0 THEN ?
                ELSE deadline_at
            END,
            lease_expires_at = ?,
            heartbeat_at = ?,
            attempt_count = attempt_count + 1,
            started_at = COALESCE(started_at, ?),
            updated_at = ?,
            completed_at = NULL,
            terminal_error_code = NULL,
            terminal_error_message = NULL,
            terminal_error_json = NULL,
            error_text = NULL
        WHERE id = ?
          AND cancel_requested = 0
          AND attempt_count < max_attempts
          AND (
              (
                  status = 'queued'
                  AND attempt_count = 0
                  AND created_at > ?
              )
              OR (
                  (deadline_at IS NULL OR deadline_at > ?)
                  AND (
                      (status = 'queued' AND attempt_count > 0)
                      OR (
                          status = 'running'
                          AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                      )
                  )
              )
          )
        """,
        (
            lease_owner,
            deadline_at,
            lease_expires_at,
            now_text,
            now_text,
            now_text,
            job_id,
            queue_cutoff,
            now_text,
            now_text,
        ),
    )
    if claimed:
        return get_job(conn, job_id)
    _fail_unclaimable_job(conn, job_id, now_text)
    return None


def lease_next_job(
    conn: Any,
    *,
    lease_owner: str,
    now: datetime | str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> dict[str, Any] | None:
    now_text = _iso(now)
    candidates = fetch_all(
        conn,
        """
        SELECT id
        FROM reel_generation_jobs
        WHERE TRIM(content_fingerprint) <> ''
          AND (
            status = 'queued'
            OR (
               status = 'running'
               AND (
                   lease_expires_at IS NULL
                   OR lease_expires_at <= ?
                   OR (deadline_at IS NOT NULL AND deadline_at <= ?)
               )
            )
          )
        ORDER BY created_at, id
        LIMIT 32
        """,
        (now_text, now_text),
    )
    for candidate in candidates:
        leased = lease_job(
            conn,
            job_id=str(candidate.get("id") or ""),
            lease_owner=lease_owner,
            now=now_text,
            lease_seconds=lease_seconds,
        )
        if leased:
            return leased
    return None


def heartbeat_job(
    conn: Any,
    *,
    job_id: str,
    lease_owner: str,
    now: datetime | str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> bool:
    if not str(lease_owner or "").strip():
        raise ValueError("lease_owner is required")
    now_dt = _utc_now(now)
    now_text = now_dt.isoformat()
    lease_duration = max(1, min(DEFAULT_LEASE_SECONDS, int(lease_seconds)))
    lease_expires_at = (now_dt + timedelta(seconds=lease_duration)).isoformat()
    return bool(
        execute_modify(
            conn,
            """
            UPDATE reel_generation_jobs
            SET heartbeat_at = ?,
                lease_expires_at = CASE
                    WHEN deadline_at IS NOT NULL AND deadline_at < ? THEN deadline_at
                    ELSE ?
                END,
                updated_at = ?
            WHERE id = ?
              AND status = 'running'
              AND lease_owner = ?
              AND cancel_requested = 0
              AND lease_expires_at > ?
              AND (deadline_at IS NULL OR deadline_at > ?)
            """,
            (
                now_text,
                lease_expires_at,
                lease_expires_at,
                now_text,
                job_id,
                lease_owner,
                now_text,
                now_text,
            ),
        )
    )


def request_cancellation(
    conn: Any,
    *,
    job_id: str,
    now: datetime | str | None = None,
) -> dict[str, Any] | None:
    """Persist cancellation, revoke any lease, and emit the terminal event once."""
    timestamp = _iso(now)
    with _atomic_write(conn):
        updated = execute_modify(
            conn,
            """
        UPDATE reel_generation_jobs
        SET status = 'cancelled',
            phase = 'terminal',
            progress = 1.0,
            cancel_requested = 1,
            cancel_requested_at = COALESCE(cancel_requested_at, ?),
            lease_owner = NULL,
            lease_expires_at = NULL,
            heartbeat_at = NULL,
            completed_at = ?,
            updated_at = ?,
            terminal_error_code = COALESCE(terminal_error_code, 'cancelled'),
            terminal_error_message = COALESCE(terminal_error_message, 'Generation cancelled.'),
            error_text = 'cancelled'
        WHERE id = ? AND status IN ('queued', 'running')
            """,
            (timestamp, timestamp, timestamp, job_id),
        )
        row = get_job(conn, job_id)
        if updated and row:
            append_event(
                conn,
                job_id=job_id,
                event_type="terminal",
                payload=_terminal_payload(row),
                now=timestamp,
            )
            row = get_job(conn, job_id)
        return row


def cancellation_requested(conn: Any, job_id: str) -> bool:
    row = fetch_one(
        conn,
        "SELECT status, cancel_requested FROM reel_generation_jobs WHERE id = ?",
        (job_id,),
    )
    return bool(row and (int(row.get("cancel_requested") or 0) or row.get("status") == "cancelled"))


def update_progress(
    conn: Any,
    *,
    job_id: str,
    lease_owner: str,
    phase: str,
    progress: float,
    model_used: str | None = None,
    quality_degraded: bool | None = None,
    usage: dict[str, Any] | None = None,
    now: datetime | str | None = None,
) -> bool:
    if not str(lease_owner or "").strip():
        raise ValueError("lease_owner is required")
    if not str(phase or "").strip():
        raise ValueError("phase is required")
    now_text = _iso(now)
    assignments = ["phase = ?", "progress = ?", "updated_at = ?"]
    values: list[Any] = [str(phase), max(0.0, min(1.0, float(progress))), now_text]
    if model_used is not None:
        assignments.append("model_used = ?")
        values.append(str(model_used))
    if quality_degraded is not None:
        assignments.append("quality_degraded = ?")
        values.append(1 if quality_degraded else 0)
    if usage is not None:
        assignments.append("usage_json = ?")
        values.append(dumps_json(usage))
    values.extend([job_id, lease_owner, now_text, now_text])
    return bool(
        execute_modify(
            conn,
            f"UPDATE reel_generation_jobs SET {', '.join(assignments)} "
            "WHERE id = ? AND status = 'running' AND lease_owner = ? "
            "AND cancel_requested = 0 AND lease_expires_at > ? "
            "AND (deadline_at IS NULL OR deadline_at > ?)",
            values,
        )
    )


def transition_terminal(
    conn: Any,
    *,
    job_id: str,
    status: JobStatus,
    result_generation_id: str | None = None,
    lease_owner: str | None = None,
    model_used: str | None = None,
    quality_degraded: bool | None = None,
    usage: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    error_detail: dict[str, Any] | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any] | None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"not a terminal generation status: {status}")
    if lease_owner is not None and not str(lease_owner).strip():
        raise ValueError("lease_owner cannot be blank")
    with _atomic_write(conn):
        existing = get_job(conn, job_id)
        if not existing:
            return None
        if str(existing.get("status") or "") in TERMINAL_STATUSES:
            return existing
        effective_status = "cancelled" if int(existing.get("cancel_requested") or 0) else status
        effective_error_code = "cancelled" if effective_status == "cancelled" else error_code
        effective_error_message = (
            "Generation cancelled." if effective_status == "cancelled" else error_message
        )
        timestamp = _iso(now)
        owner_clause = ""
        params: list[Any] = [
            effective_status,
            result_generation_id,
            model_used,
            1
            if quality_degraded
            else 0
            if quality_degraded is not None
            else int(existing.get("quality_degraded") or 0),
            dumps_json(usage)
            if usage is not None
            else str(existing.get("usage_json") or "{}"),
            effective_error_code,
            effective_error_message,
            dumps_json(error_detail) if error_detail is not None else None,
            effective_error_message,
            timestamp,
            timestamp,
            job_id,
        ]
        if lease_owner is not None:
            owner_clause = (
                " AND lease_owner = ? AND cancel_requested = 0"
                " AND lease_expires_at > ?"
                " AND (deadline_at IS NULL OR deadline_at > ?)"
            )
            params.extend([lease_owner, timestamp, timestamp])
        updated = execute_modify(
            conn,
            """
        UPDATE reel_generation_jobs
        SET status = ?,
            phase = 'terminal',
            progress = 1.0,
            result_generation_id = COALESCE(?, result_generation_id),
            model_used = COALESCE(?, model_used),
            quality_degraded = ?,
            usage_json = ?,
            terminal_error_code = ?,
            terminal_error_message = ?,
            terminal_error_json = ?,
            error_text = ?,
            lease_owner = NULL,
            lease_expires_at = NULL,
            heartbeat_at = NULL,
            completed_at = ?,
            updated_at = ?
        WHERE id = ? AND status IN ('queued', 'running')
            """
            + owner_clause,
            params,
        )
        if not updated:
            current = get_job(conn, job_id)
            if lease_owner is not None and str((current or {}).get("status") or "") in ACTIVE_STATUSES:
                raise JobLeaseLostError(f"generation job lease is no longer active: {job_id}")
            return current
        row = get_job(conn, job_id)
        if row:
            append_event(
                conn,
                job_id=job_id,
                event_type="terminal",
                payload=_terminal_payload(row),
                now=timestamp,
            )
        return get_job(conn, job_id)


def record_provider_usage(
    conn: Any,
    *,
    job_id: str,
    provider: str,
    operation: str,
    model: str | None = None,
    billable_requests: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    now: datetime | str | None = None,
    usage_id: str | None = None,
) -> str:
    record_id = str(usage_id or uuid.uuid4())
    safe_input = max(0, int(input_tokens or 0))
    safe_output = max(0, int(output_tokens or 0))
    insert(
        conn,
        "generation_provider_usage",
        {
            "id": record_id,
            "job_id": job_id,
            "provider": str(provider),
            "operation": str(operation),
            "model": str(model) if model else None,
            "billable_requests": max(0, int(billable_requests or 0)),
            "input_tokens": safe_input,
            "output_tokens": safe_output,
            "total_tokens": max(0, int(total_tokens if total_tokens is not None else safe_input + safe_output)),
            "metadata_json": dumps_json(metadata or {}),
            "created_at": _iso(now),
        },
    )
    return record_id


__all__ = [
    "ACTIVE_STATUSES",
    "ALL_STATUSES",
    "DEFAULT_DEADLINE_SECONDS",
    "DEFAULT_HEARTBEAT_SECONDS",
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_QUEUE_TTL_SECONDS",
    "EVENT_TYPES",
    "GenerationQueueFullError",
    "JobLeaseLostError",
    "TERMINAL_STATUSES",
    "append_event",
    "build_request_key",
    "cancellation_requested",
    "expire_stale_queued_job",
    "find_active_job",
    "find_completed_job",
    "get_job",
    "heartbeat_job",
    "lease_job",
    "lease_next_job",
    "material_content_fingerprint",
    "record_provider_usage",
    "replay_events",
    "request_cancellation",
    "submit_or_get_active",
    "transition_terminal",
    "update_progress",
]
