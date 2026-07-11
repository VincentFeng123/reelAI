from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from fastapi.responses import JSONResponse

from backend.app import db
from backend.app import main
from backend.app.models import ReelsGenerateRequest
from backend.app.ingestion.models import IngestRequest
from backend.app.services import generation_jobs


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    db._migrate_durable_generation_foundation_sqlite(conn)
    conn.execute(
        "INSERT INTO materials (id, subject_tag, raw_text, source_type, created_at) "
        "VALUES ('m1', 'cell biology', 'Cell biology', 'topic', '2026-07-10T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO concepts (id, material_id, title, keywords_json, summary, created_at) "
        "VALUES ('c1', 'm1', 'Mitochondria', '[]', 'Cell energy', '2026-07-10T00:00:00+00:00')"
    )
    return conn


def _patch_request_context(monkeypatch, conn: sqlite3.Connection) -> None:
    @contextmanager
    def connection(**_kwargs):
        yield conn

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "_require_community_client_identity", lambda _request: "owner")
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_resolve_learner_identity", lambda *_args, **_kwargs: "learner-1")
    monkeypatch.setattr(
        main.reel_service,
        "learner_progress",
        lambda *_args, **_kwargs: {"selected_level": "beginner", "global_adjustment": 0.0},
    )


def test_generation_worker_pool_executes_two_jobs_concurrently_with_distinct_owners(
    monkeypatch,
) -> None:
    main._stop_generation_worker()
    entered = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    leased_owners: set[str] = set()
    running_jobs: list[tuple[str, str]] = []

    @contextmanager
    def connection(**_kwargs):
        yield object()

    def lease_next(_conn, *, lease_owner: str, lease_seconds: int):
        del lease_seconds
        with state_lock:
            if lease_owner in leased_owners:
                return None
            leased_owners.add(lease_owner)
            return {"id": f"job-{len(leased_owners)}", "lease_owner": lease_owner}

    def run_job(job_row: dict, _worker_stop: threading.Event) -> None:
        with state_lock:
            running_jobs.append((str(job_row["id"]), str(job_row["lease_owner"])))
            if len(running_jobs) == 2:
                entered.set()
        assert release.wait(timeout=2.0)

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "lease_next_job", lease_next)
    monkeypatch.setattr(main, "_run_leased_generation_job", run_job)

    try:
        main._start_generation_worker()
        assert entered.wait(timeout=2.0)
        assert len(main._generation_worker_threads) == main.GENERATION_WORKER_COUNT == 2
        worker_threads = list(main._generation_worker_threads)
        main._start_generation_worker()
        assert main._generation_worker_threads == worker_threads
        assert len({owner for _job_id, owner in running_jobs}) == 2
        assert all(owner.startswith("worker-") for _job_id, owner in running_jobs)
        assert 0 < main.GENERATION_HEARTBEAT_SEC < main.GENERATION_LEASE_SEC
        assert main.GENERATION_WORKER_POLL_SEC > 0
        health = main.admin_health()
        assert health["generation_worker_alive"] is True
        assert health["generation_worker_count"] == 2
        assert health["generation_workers_alive"] == 2
    finally:
        release.set()
        main._stop_generation_worker()
    assert main._generation_worker_threads == []


def test_generation_worker_stop_retains_live_pool_and_restart_uses_fresh_state(
    monkeypatch,
) -> None:
    main._stop_generation_worker()

    class FakeThread:
        instances: list["FakeThread"] = []

        def __init__(self, *, target, args, name, daemon) -> None:
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon
            self.alive = False
            self.__class__.instances.append(self)

        def start(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout=None) -> None:
            del timeout

    monkeypatch.setattr(main.threading, "Thread", FakeThread)

    main._start_generation_worker()
    first_threads = list(main._generation_worker_threads)
    first_stop = main._generation_worker_stop
    first_owners = {str(thread.args[0]) for thread in first_threads}
    assert len(first_threads) == main.GENERATION_WORKER_COUNT

    main._stop_generation_worker()
    assert first_stop.is_set()
    assert main._generation_worker_threads == first_threads

    main._start_generation_worker()
    assert main._generation_worker_threads == first_threads
    assert len(FakeThread.instances) == main.GENERATION_WORKER_COUNT

    for thread in first_threads:
        thread.alive = False
    main._start_generation_worker()
    second_threads = list(main._generation_worker_threads)
    second_owners = {str(thread.args[0]) for thread in second_threads}
    assert second_threads != first_threads
    assert main._generation_worker_stop is not first_stop
    assert first_owners.isdisjoint(second_owners)

    for thread in second_threads:
        thread.alive = False
    main._stop_generation_worker()
    assert main._generation_worker_threads == []


def test_admin_health_requires_the_full_worker_pool_but_basic_health_stays_compatible(
    monkeypatch,
) -> None:
    class StateThread:
        def __init__(self, alive: bool) -> None:
            self.alive = alive

        def is_alive(self) -> bool:
            return self.alive

    monkeypatch.setattr(
        main,
        "_generation_worker_threads",
        [StateThread(True), StateThread(False)],
    )

    admin = main.admin_health()
    assert admin["ok"] is False
    assert admin["generation_worker_alive"] is False
    assert admin["generation_worker_count"] == 2
    assert admin["generation_workers_alive"] == 1
    assert main.health() == {"ok": True}


def test_generation_worker_db_probe_stops_for_every_invalid_lease_state(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="worker-probe-request",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={},
        now="2026-07-10T12:00:00+00:00",
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-a",
        now="2026-07-10T12:00:00+00:00",
    )
    assert leased
    checked_at = datetime(2026, 7, 10, 12, 0, 1, tzinfo=timezone.utc)
    try:
        assert not main._generation_job_db_should_stop(
            job["id"], "worker-a", now=checked_at
        )
        assert main._generation_job_db_should_stop("missing", "worker-a", now=checked_at)

        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'queued' WHERE id = ?", (job["id"],)
        )
        assert main._generation_job_db_should_stop(job["id"], "worker-a", now=checked_at)

        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'running', lease_owner = 'worker-b' WHERE id = ?",
            (job["id"],),
        )
        assert main._generation_job_db_should_stop(job["id"], "worker-a", now=checked_at)

        conn.execute(
            "UPDATE reel_generation_jobs SET lease_owner = 'worker-a', lease_expires_at = ? WHERE id = ?",
            ((checked_at - timedelta(seconds=1)).isoformat(), job["id"]),
        )
        assert main._generation_job_db_should_stop(job["id"], "worker-a", now=checked_at)

        conn.execute(
            "UPDATE reel_generation_jobs SET lease_expires_at = ?, deadline_at = ? WHERE id = ?",
            (
                (checked_at + timedelta(seconds=30)).isoformat(),
                checked_at.isoformat(),
                job["id"],
            ),
        )
        assert main._generation_job_db_should_stop(job["id"], "worker-a", now=checked_at)
    finally:
        conn.close()


def test_generation_worker_reuses_attached_generation_after_lease_reclaim(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="lease-reclaim-generation",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "slow", "num_reels": 2},
        now=now,
    )
    first_lease = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-a",
        now=now,
    )
    assert first_lease
    generation_ids: list[str] = []

    def yield_after_progress(worker_conn, **kwargs) -> None:
        generation_id = str(kwargs["generation_id"])
        generation_ids.append(generation_id)
        if len(generation_ids) == 1:
            worker_conn.execute(
                "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
                "VALUES ('lease-video', 'Lease video', 'Test', 120, ?)",
                (now.isoformat(),),
            )
            worker_conn.execute(
                "INSERT INTO reels "
                "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                "transcript_snippet, takeaways_json, base_score, generation_id, created_at) "
                "VALUES ('lease-reel', 'm1', 'c1', 'lease-video', '', 0, 30, '', '[]', 1, ?, ?)",
                (generation_id, now.isoformat()),
            )
            worker_conn.execute(
                "UPDATE reel_generation_jobs SET lease_owner = 'worker-stolen' WHERE id = ?",
                (job["id"],),
            )
            return
        raise main.GenerationCancelledError("worker yielded")

    monkeypatch.setattr(main.reel_service, "generate_reels", yield_after_progress)
    try:
        main._run_leased_generation_job(first_lease, threading.Event())
        after_first = generation_jobs.get_job(conn, job["id"])
        assert after_first and after_first["result_generation_id"] == generation_ids[0]
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_heads").fetchone()[0] == 0

        reclaim_at = now + timedelta(seconds=1)
        conn.execute(
            "UPDATE reel_generation_jobs SET lease_expires_at = ? WHERE id = ?",
            ((reclaim_at - timedelta(seconds=1)).isoformat(), job["id"]),
        )
        second_lease = generation_jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-b",
            now=reclaim_at,
        )
        assert second_lease
        main._run_leased_generation_job(second_lease, threading.Event())

        assert generation_ids == [generation_ids[0], generation_ids[0]]
        assert conn.execute("SELECT COUNT(*) FROM reel_generations").fetchone()[0] == 1
        persisted = conn.execute(
            "SELECT generation_id FROM reels WHERE id = 'lease-reel'"
        ).fetchone()
        assert persisted[0] == generation_ids[0]
    finally:
        conn.close()


def test_status_and_stream_terminalize_stale_queue_without_duplicate_events(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    stale_time = datetime.now(timezone.utc) - timedelta(
        seconds=generation_jobs.DEFAULT_QUEUE_TTL_SECONDS + 1
    )
    status_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="stale-status-request",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={},
        now=stale_time,
    )
    stream_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="stale-stream-request",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={},
        now=stale_time,
    )

    try:
        status = main.generation_status(object(), status_job["id"])
        assert status["status"] == "failed"
        assert status["error"]["code"] == "queue_timeout"

        async def collect() -> list[dict]:
            response = await main.generation_stream(object(), stream_job["id"])
            rows: list[dict] = []
            async for chunk in response.body_iterator:
                rows.append(json.loads(chunk))
            return rows

        events = asyncio.run(collect())
        assert [event["type"] for event in events] == ["terminal"]
        assert events[0]["payload"]["error"]["code"] == "queue_timeout"
        assert len(generation_jobs.replay_events(conn, job_id=status_job["id"])) == 1
        assert len(generation_jobs.replay_events(conn, job_id=stream_job["id"])) == 1
    finally:
        conn.close()


def test_generate_returns_202_and_idempotently_reuses_active_job(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    payload = ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=3)
    try:
        first = asyncio.run(main.generate_reels(object(), payload))
        second = asyncio.run(main.generate_reels(object(), payload))
        assert isinstance(first, JSONResponse)
        assert isinstance(second, JSONResponse)
        assert first.status_code == second.status_code == 202
        first_body = json.loads(first.body)
        second_body = json.loads(second.body)
        assert first_body["job_id"] == second_body["job_id"]
        assert first_body["stream_url"].endswith(first_body["job_id"])
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs WHERE status IN ('queued', 'running')"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_generate_rejects_multi_platform_before_submitting(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    payload = ReelsGenerateRequest(material_id="m1", multi_platform_search=True)
    try:
        try:
            asyncio.run(main.generate_reels(object(), payload))
        except main.HTTPException as exc:
            assert exc.status_code == 422
            assert exc.detail["code"] == "unsupported_retrieval_platform"
        else:  # pragma: no cover
            raise AssertionError("multi-platform request was accepted")
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 0
    finally:
        conn.close()


def test_ingestion_rejects_multi_platform_before_provider_work(monkeypatch) -> None:
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    payload = IngestRequest(
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        multi_platform_search=True,
    )
    try:
        asyncio.run(main.ingest_url_endpoint(object(), payload))
    except main.HTTPException as exc:
        assert exc.status_code == 422
        assert exc.detail["code"] == "unsupported_retrieval_platform"
    else:  # pragma: no cover
        raise AssertionError("multi-platform ingestion request was accepted")


def test_generation_stream_replays_monotonic_persisted_events(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="request-key",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast"},
    )
    leased = generation_jobs.lease_job(conn, job_id=job["id"], lease_owner="worker")
    assert leased
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="candidate",
        payload={"reel": {"reel_id": "provisional"}, "provisional": True},
        lease_owner="worker",
    )
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="final",
        payload={"reels": [], "authoritative": True},
        lease_owner="worker",
    )
    generation_jobs.transition_terminal(
        conn,
        job_id=job["id"],
        status="exhausted",
        lease_owner="worker",
        error_code="inventory_exhausted",
        error_message="No inventory.",
    )

    async def collect() -> list[dict]:
        response = await main.generation_stream(object(), job["id"], after_seq=0)
        rows: list[dict] = []
        async for chunk in response.body_iterator:
            rows.append(json.loads(chunk))
        return rows

    try:
        events = asyncio.run(collect())
        assert [event["type"] for event in events] == ["candidate", "final", "terminal"]
        assert [event["seq"] for event in events] == [1, 2, 3]
        assert all(event["job_id"] == job["id"] and event["timestamp"] for event in events)
    finally:
        conn.close()


def test_preflight_uses_one_metadata_search_and_no_generation(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)

    class EmptyCache:
        def get_search(self, _key):
            return None

    provider_calls: list[str] = []
    monkeypatch.setattr(main, "DatabaseProviderCache", lambda: EmptyCache())
    monkeypatch.setattr(
        main,
        "supadata_search_one",
        lambda query, *_args, **_kwargs: provider_calls.append(query) or {
            "videos": [{"id": "dQw4w9WgXcQ"}],
            "cache_hit": False,
            "evidence_age_sec": 0,
            "filters_applied": {"features": []},
        },
    )
    monkeypatch.setattr(
        main.reel_service,
        "generate_reels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("preflight called generation")),
    )
    try:
        result = main.can_generate_reels(
            object(),
            ReelsGenerateRequest(material_id="m1", concept_id="c1"),
        )
        assert result["availability"] == "available"
        assert result["candidate_count"] == 1
        assert len(provider_calls) == 1
    finally:
        conn.close()


def test_feed_autofill_false_never_submits_and_true_submits_before_return(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_ranked_request_reels", lambda *_args, **_kwargs: [])
    try:
        empty = main.feed(object(), material_id="m1", autofill=False)
        assert empty["reels"] == []
        assert empty["generation_job_id"] is None
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 0

        queued = main.feed(object(), material_id="m1", autofill=True)
        assert queued["reels"] == []
        assert queued["generation_job_id"]
        assert queued["generation_job_status"] == "queued"
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 1
    finally:
        conn.close()


def test_feed_rejects_deprecated_multi_platform_true(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    try:
        try:
            main.feed(object(), material_id="m1", multi_platform_search=True)
        except main.HTTPException as exc:
            assert exc.status_code == 422
            assert exc.detail["code"] == "unsupported_retrieval_platform"
        else:  # pragma: no cover
            raise AssertionError("multi-platform feed request was accepted")
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 0
    finally:
        conn.close()
