from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from fastapi.responses import JSONResponse
import pytest

from backend.app import db
from backend.app import main
from backend.app.clip_engine.errors import ProviderQuotaError, ProviderTransientError
from backend.app.models import ReelsGenerateRequest
from backend.app.ingestion.models import IngestRequest
from backend.app.services import generation_jobs


def test_generate_request_supports_full_material_inventory() -> None:
    assert ReelsGenerateRequest(material_id="m1").num_reels == 20
    assert ReelsGenerateRequest(material_id="m1", num_reels=300).num_reels == 300
    with pytest.raises(ValueError):
        ReelsGenerateRequest(material_id="m1", num_reels=301)


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


def _insert_generation_reel(
    conn: sqlite3.Connection,
    *,
    generation_id: str,
    reel_id: str,
    video_id: str,
    created_at: str,
) -> dict:
    conn.execute(
        "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
        "VALUES (?, ?, 'Test', 120, ?)",
        (video_id, video_id, created_at),
    )
    conn.execute(
        "INSERT INTO reels "
        "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
        "transcript_snippet, takeaways_json, base_score, generation_id, created_at) "
        "VALUES (?, 'm1', 'c1', ?, '', 0, 30, '', '[]', 1, ?, ?)",
        (reel_id, video_id, generation_id, created_at),
    )
    return {
        "reel_id": reel_id,
        "video_id": video_id,
        "t_start": 0.0,
        "t_end": 30.0,
    }


def _set_reel_boundary_state(
    conn: sqlite3.Connection,
    *,
    reel_id: str,
    boundary_status: str,
    surface_eligible: bool = True,
) -> None:
    conn.execute(
        "UPDATE reels SET search_context_json = ? WHERE id = ?",
        (
            json.dumps({
                "surface_eligible": surface_eligible,
                "boundary_status": boundary_status,
                "selection_contract_version": "confidence_v1",
                "directly_teaches_topic": True,
                "substantive": True,
            }),
            reel_id,
        ),
    )


def _terminal_job_for_generation(
    conn: sqlite3.Connection,
    *,
    request_key: str,
    generation_id: str,
    completed_at: str,
    learner_id: str = "learner-1",
    status: str = "completed",
    concept_id: str | None = "c1",
) -> dict:
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id=concept_id,
        request_key=request_key,
        content_fingerprint="prior-fingerprint",
        learner_id=learner_id,
        request_params={"generation_mode": "slow", "num_reels": 12},
    )
    conn.execute(
        "UPDATE reel_generation_jobs SET status = ?, phase = 'terminal', "
        "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
        "WHERE id = ?",
        (status, generation_id, completed_at, completed_at, job["id"]),
    )
    return {**job, "status": status, "result_generation_id": generation_id}


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


def test_generation_worker_propagates_the_full_source_generation_chain(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    root_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="source-root",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="source-parent",
        generation_mode="slow",
        retrieval_profile="unified",
        source_generation_id=root_generation_id,
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="source-chain-worker",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "slow", "num_reels": 1},
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-source-chain",
        now=now,
    )
    assert leased
    generation_calls: list[dict] = []

    def create_one_reel(worker_conn, **kwargs) -> None:
        generation_calls.append(kwargs)
        generation_context = kwargs["generation_context"]
        generation_context.record_gemini(
            attempt=1,
            model_used="gemini-primary",
            quality_degraded=False,
            usage={"prompt_tokens": 10, "candidate_tokens": 2, "total_tokens": 12},
        )
        generation_context.record_cache_hit(
            provider="gemini",
            operation="segmentation",
        )
        worker_conn.execute(
            "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
            "VALUES ('source-chain-video', 'Source chain video', 'Test', 120, ?)",
            (now.isoformat(),),
        )
        worker_conn.execute(
            "INSERT INTO reels "
            "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
            "transcript_snippet, takeaways_json, base_score, generation_id, created_at) "
            "VALUES ('source-chain-reel', 'm1', 'c1', 'source-chain-video', '', "
            "0, 30, '', '[]', 1, ?, ?)",
            (str(kwargs["generation_id"]), now.isoformat()),
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", create_one_reel)
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [{"reel_id": "source-chain-reel"}],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed_job = generation_jobs.get_job(conn, job["id"])
        assert completed_job and completed_job["status"] == "completed"
        assert completed_job["model_used"] == "gemini-primary"
        result_generation_id = str(completed_job["result_generation_id"])
        result_generation = conn.execute(
            "SELECT source_generation_id FROM reel_generations WHERE id = ?",
            (result_generation_id,),
        ).fetchone()
        assert result_generation["source_generation_id"] == source_generation_id
        assert len(generation_calls) == 1
        assert generation_calls[0]["exclude_generation_ids"] == [
            root_generation_id,
            source_generation_id,
        ]
    finally:
        conn.close()


def test_cross_request_source_count_leaves_two_slots_for_fresh_bootstrap(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="prior-level-request",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="new-level-request",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "slow", "num_reels": 5},
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-cross-level",
        now=now,
    )
    assert leased
    generated_count = 0
    calls: list[dict] = []

    def count_generation_reels(_conn, generation_id: str) -> int:
        return 10 if generation_id == source_generation_id else generated_count

    def generate_stage(_worker_conn, **kwargs) -> None:
        nonlocal generated_count
        calls.append(kwargs)
        generated_count += int(kwargs["max_new_reels"])

    monkeypatch.setattr(main, "_count_generation_reels", count_generation_reels)
    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": f"reel-{index}"} for index in range(5)
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert len(calls) == 1
        assert calls[0]["retrieval_profile"] == "bootstrap"
        assert calls[0]["max_new_reels"] == 2
        assert generation_jobs.get_job(conn, job["id"])["status"] == "completed"
    finally:
        conn.close()


def test_slow_generation_bootstraps_then_deepens_with_shared_caps(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="two-stage-source",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="source-reel",
        video_id="source-video",
        created_at=now.isoformat(),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="two-stage-slow",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "slow",
            "num_reels": 4,
            "exclude_video_ids": ["manual-exclusion"],
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-two-stage",
        now=now,
    )
    assert leased
    calls: list[dict] = []
    progress_updates: list[tuple[str, float]] = []
    update_progress = main.update_generation_progress

    def capture_progress(*args, **kwargs):
        progress_updates.append((str(kwargs["phase"]), float(kwargs["progress"])))
        return update_progress(*args, **kwargs)

    def generate_stage(worker_conn, **kwargs) -> None:
        calls.append(kwargs)
        profile = kwargs["retrieval_profile"]
        analyzed = kwargs["analyzed_video_ids"]
        if profile == "bootstrap":
            analyzed.update({"bootstrap-video", "bootstrap-empty-video"})
            kwargs["retrieved_video_ids"].update(
                {
                    "bootstrap-video",
                    "bootstrap-empty-video",
                    "bootstrap-retrieved-only-video",
                }
            )
            staged = [("bootstrap-reel", "bootstrap-video")]
        else:
            analyzed.update({"deep-video-1", "deep-video-2"})
            staged = [
                ("deep-reel-1", "deep-video-1"),
                ("deep-reel-2", "deep-video-2"),
            ]
        for reel_id, video_id in staged[: kwargs["max_new_reels"]]:
            reel = _insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id=reel_id,
                video_id=video_id,
                created_at=now.isoformat(),
            )
            kwargs["on_reel_created"](reel)

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(main, "update_generation_progress", capture_progress)
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": "deep-reel-2"},
            {"reel_id": "source-reel"},
            {"reel_id": "deep-reel-1"},
            {"reel_id": "bootstrap-reel"},
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert [call["retrieval_profile"] for call in calls] == ["bootstrap", "deep"]
        assert [call["max_new_reels"] for call in calls] == [2, 2]
        assert calls[0]["max_generation_videos"] == 3
        assert calls[1]["max_generation_videos"] == 3
        assert set(calls[1]["exclude_video_ids"]) == {
            "manual-exclusion",
            "bootstrap-video",
            "bootstrap-empty-video",
            "bootstrap-retrieved-only-video",
        }
        assert ("retrieval", 0.45) in progress_updates
        assert ("ranking", 0.85) in progress_updates
        events = generation_jobs.replay_events(conn, job_id=job["id"])
        assert [event["type"] for event in events] == [
            "candidate",
            "candidate",
            "candidate",
            "final",
            "terminal",
        ]
        assert [
            event["payload"]["reel"]["reel_id"]
            for event in events
            if event["type"] == "candidate"
        ] == ["bootstrap-reel", "deep-reel-1", "deep-reel-2"]
        final_reel_ids = [
            reel["reel_id"]
            for event in events
            if event["type"] == "final"
            for reel in event["payload"]["reels"]
        ]
        assert final_reel_ids == [
            "deep-reel-2",
            "source-reel",
            "deep-reel-1",
            "bootstrap-reel",
        ]
        assert "bootstrap-reel" in final_reel_ids
        assert conn.execute(
            "SELECT COUNT(*) FROM reels WHERE generation_id = ?",
            (str((generation_jobs.get_job(conn, job["id"]) or {})["result_generation_id"]),),
        ).fetchone()[0] == 3
    finally:
        conn.close()


def test_fast_generation_stops_after_two_clip_bootstrap(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="two-stage-fast",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 4},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-fast-stage",
        now=now,
    )
    assert leased
    calls: list[dict] = []

    def generate_bootstrap(worker_conn, **kwargs) -> None:
        calls.append(kwargs)
        kwargs["analyzed_video_ids"].update({"fast-video-1", "fast-video-2"})
        for index in range(kwargs["max_new_reels"]):
            reel = _insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id=f"fast-reel-{index}",
                video_id=f"fast-video-{index + 1}",
                created_at=now.isoformat(),
            )
            kwargs["on_reel_created"](reel)

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_bootstrap)
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": "fast-reel-0"},
            {"reel_id": "fast-reel-1"},
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert len(calls) == 1
        assert calls[0]["retrieval_profile"] == "bootstrap"
        assert calls[0]["max_generation_videos"] == 3
        assert calls[0]["max_new_reels"] == 2
        events = generation_jobs.replay_events(conn, job_id=job["id"])
        assert [event["type"] for event in events] == [
            "candidate",
            "candidate",
            "final",
            "terminal",
        ]
        final_event = next(event for event in events if event["type"] == "final")
        terminal_event = next(event for event in events if event["type"] == "terminal")
        assert len(final_event["payload"]["reels"]) == 2
        assert terminal_event["payload"]["status"] == "partial"
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("mode", "expected_profiles", "expected_status"),
    [
        ("slow", ["bootstrap", "deep"], "completed"),
        ("fast", ["bootstrap"], "exhausted"),
    ],
)
def test_bootstrap_deadline_exhaustion_is_a_stage_result(
    monkeypatch,
    mode: str,
    expected_profiles: list[str],
    expected_status: str,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"bootstrap-deadline-{mode}",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": mode, "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner=f"worker-bootstrap-deadline-{mode}",
        now=now,
    )
    assert leased
    calls: list[dict] = []

    def generate_stage(worker_conn, **kwargs) -> None:
        profile = str(kwargs["retrieval_profile"])
        calls.append(kwargs)
        if profile == "bootstrap":
            kwargs["analyzed_video_ids"].add("bootstrap-analyzed-timeout")
            kwargs["retrieved_video_ids"].add("bootstrap-retrieved-timeout")
            raise ProviderTransientError(
                "Supadata search timed out.",
                provider="supadata",
                operation="search",
                detail="generation deadline exceeded",
            )
        _insert_generation_reel(
            worker_conn,
            generation_id=str(kwargs["generation_id"]),
            reel_id="deep-after-timeout",
            video_id="deep-video-after-timeout",
            created_at=now.isoformat(),
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    if mode == "slow":
        monkeypatch.setattr(
            main,
            "_generation_job_reels",
            lambda *_args, **_kwargs: [{"reel_id": "deep-after-timeout"}],
        )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed_job = generation_jobs.get_job(conn, job["id"])
        assert completed_job and completed_job["status"] == expected_status
        assert [call["retrieval_profile"] for call in calls] == expected_profiles
        if mode == "slow":
            assert set(calls[1]["exclude_video_ids"]) == {
                "bootstrap-retrieved-timeout"
            }
        terminal_error_code = conn.execute(
            "SELECT terminal_error_code FROM reel_generation_jobs WHERE id = ?",
            (job["id"],),
        ).fetchone()[0]
        assert terminal_error_code == (
            "inventory_exhausted" if mode == "fast" else None
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "provider_error",
    [
        ProviderQuotaError(
            "Supadata quota is exhausted.",
            provider="supadata",
            operation="search",
        ),
        ProviderTransientError(
            "Could not reach Supadata search.",
            provider="supadata",
            operation="search",
            detail="connection reset",
        ),
    ],
)
def test_bootstrap_non_deadline_provider_failures_remain_fatal(
    monkeypatch,
    provider_error,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"bootstrap-fatal-{provider_error.code}-{provider_error.detail}",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "slow", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-bootstrap-fatal",
        now=now,
    )
    assert leased
    profiles: list[str] = []

    def fail_stage(_worker_conn, **kwargs) -> None:
        profiles.append(str(kwargs["retrieval_profile"]))
        raise provider_error

    monkeypatch.setattr(main.reel_service, "generate_reels", fail_stage)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        failed_job = generation_jobs.get_job(conn, job["id"])
        assert failed_job and failed_job["status"] == "failed"
        assert profiles == ["bootstrap"]
        assert conn.execute(
            "SELECT terminal_error_code FROM reel_generation_jobs WHERE id = ?",
            (job["id"],),
        ).fetchone()[0] == provider_error.code
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


def test_generate_top_up_uses_completed_result_as_the_new_job_source(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    payload = ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=5)
    try:
        first = asyncio.run(main.generate_reels(object(), payload))
        first_job_id = json.loads(first.body)["job_id"]
        first_job = generation_jobs.get_job(conn, first_job_id)
        assert first_job
        source_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(first_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'completed', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (source_generation_id, completed_at, completed_at, first_job_id),
        )
        monkeypatch.setattr(
            main,
            "_generation_job_reels",
            lambda *_args, **_kwargs: [{"reel_id": "only-existing-reel"}],
        )

        top_up = asyncio.run(main.generate_reels(object(), payload))

        assert isinstance(top_up, JSONResponse)
        assert top_up.status_code == 202
        top_up_job_id = json.loads(top_up.body)["job_id"]
        assert top_up_job_id != first_job_id
        top_up_job = generation_jobs.get_job(conn, top_up_job_id)
        assert top_up_job and top_up_job["status"] == "queued"
        assert top_up_job["source_generation_id"] == source_generation_id
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


def test_authoritative_job_inventory_retains_every_streamed_candidate(monkeypatch) -> None:
    conn = _conn()
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="candidate-retention",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "slow", "num_reels": 4},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="candidate-retention-worker",
        now=now,
    )
    assert leased
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="candidate",
        payload={
            "reel": {
                "reel_id": "streamed-reel",
                "video_id": "streamed-video",
                "t_start": 10.0,
                "t_end": 40.0,
            },
            "provisional": True,
        },
        lease_owner="candidate-retention-worker",
        now=now,
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [{"reel_id": "ranked-reel"}],
    )
    job_row = {
        **leased,
        "result_generation_id": "generation-1",
    }

    reels = main._generation_job_reels(conn, job_row)

    assert [reel["reel_id"] for reel in reels] == [
        "ranked-reel",
        "streamed-reel",
    ]
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


def test_feed_reuses_verified_prior_level_inventory_and_still_queues_bootstrap(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    completed_at = datetime.now(timezone.utc).isoformat()
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="prior-level-request",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="verified-prior-reel",
        video_id="verified-prior-video",
        created_at=completed_at,
    )
    _set_reel_boundary_state(
        conn,
        reel_id="verified-prior-reel",
        boundary_status="verified",
    )
    _terminal_job_for_generation(
        conn,
        request_key="prior-level-request",
        generation_id=source_generation_id,
        completed_at=completed_at,
        concept_id=None,
    )
    ranked_generation_ids: list[str | None] = []

    def ranked(*_args, **kwargs):
        ranked_generation_ids.append(kwargs.get("generation_id"))
        return [
            {"reel_id": f"verified-prior-reel-{index}"}
            for index in range(12)
        ]

    monkeypatch.setattr(main, "_ranked_request_reels", ranked)
    try:
        response = main.feed(object(), material_id="m1", autofill=True)

        assert response["generation_id"] == source_generation_id
        assert len(response["reels"]) == 5
        assert ranked_generation_ids == [source_generation_id]
        queued = conn.execute(
            "SELECT * FROM reel_generation_jobs WHERE status = 'queued' "
            "ORDER BY created_at DESC, id DESC LIMIT 1"
        ).fetchone()
        assert queued is not None
        assert queued["request_key"] != "prior-level-request"
        assert queued["source_generation_id"] == source_generation_id
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("concept_id", "expected_clause", "expected_params"),
    [
        (None, "AND concept_id IS NULL", ("m1", "learner-1", "new-request")),
        ("c1", "AND concept_id = ?", ("m1", "learner-1", "new-request", "c1")),
    ],
)
def test_cross_request_source_query_avoids_untyped_null_parameters(
    monkeypatch,
    concept_id: str | None,
    expected_clause: str,
    expected_params: tuple[str, ...],
) -> None:
    observed: dict[str, object] = {}

    def record_candidate_query(_conn, sql, params):
        observed["sql"] = sql
        observed["params"] = params
        return None

    monkeypatch.setattr(main, "fetch_one", record_candidate_query)

    result = main._verified_cross_request_source_generation(
        object(),
        material_id="m1",
        learner_id="learner-1",
        request_key="new-request",
        concept_id=concept_id,
    )

    assert result is None
    assert expected_clause in str(observed["sql"])
    assert "? IS NULL" not in str(observed["sql"])
    assert observed["params"] == expected_params


def test_cross_level_reservoir_rejects_unverified_or_implicit_surface_rows() -> None:
    conn = _conn()
    completed_at = datetime.now(timezone.utc).isoformat()
    root_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="prior-root",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    child_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="prior-child",
        generation_mode="slow",
        retrieval_profile="unified",
        source_generation_id=root_generation_id,
    )
    _insert_generation_reel(
        conn,
        generation_id=root_generation_id,
        reel_id="verified-root-reel",
        video_id="verified-root-video",
        created_at=completed_at,
    )
    _set_reel_boundary_state(
        conn,
        reel_id="verified-root-reel",
        boundary_status="verified",
    )
    _insert_generation_reel(
        conn,
        generation_id=child_generation_id,
        reel_id="legacy-child-reel",
        video_id="legacy-child-video",
        created_at=completed_at,
    )
    _terminal_job_for_generation(
        conn,
        request_key="prior-child",
        generation_id=child_generation_id,
        completed_at=completed_at,
        concept_id=None,
        status="partial",
    )
    try:
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="new-level-request",
            concept_id=None,
        ) is None

        _set_reel_boundary_state(
            conn,
            reel_id="legacy-child-reel",
            boundary_status="caption_aligned",
        )
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="new-level-request",
            concept_id=None,
        ) is None
    finally:
        conn.close()


def test_generate_job_reuses_verified_prior_level_for_the_same_concept(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    completed_at = datetime.now(timezone.utc).isoformat()
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="prior-concept-level",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="verified-concept-reel",
        video_id="verified-concept-video",
        created_at=completed_at,
    )
    _set_reel_boundary_state(
        conn,
        reel_id="verified-concept-reel",
        boundary_status="verified",
    )
    _terminal_job_for_generation(
        conn,
        request_key="prior-concept-level",
        generation_id=source_generation_id,
        completed_at=completed_at,
    )
    try:
        response = asyncio.run(main.generate_reels(
            object(),
            ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=5),
        ))
        queued = generation_jobs.get_job(conn, json.loads(response.body)["job_id"])
        assert queued is not None
        assert queued["source_generation_id"] == source_generation_id
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
