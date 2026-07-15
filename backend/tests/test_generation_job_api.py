from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.responses import JSONResponse
import pytest

from backend.app import db
from backend.app import main
from backend.app.clip_engine import segment_cache
from backend.app.clip_engine import silence as clip_engine_silence
from backend.app.clip_engine.errors import ProviderQuotaError, ProviderTransientError
from backend.app.ingestion import pipeline as ingestion_pipeline
from backend.app.ingestion.models import (
    IngestMetadata,
    IngestRequest,
    IngestSegment,
    IngestTranscriptCue,
    YouTubeSourceRef,
)
from backend.app.models import ReelOut, ReelsGenerateRequest
from backend.app.services import generation_jobs
from backend.app.services.reels import ReelService


def test_generate_request_supports_full_material_inventory() -> None:
    assert ReelsGenerateRequest(material_id="m1").num_reels == 20
    assert ReelsGenerateRequest(material_id="m1", num_reels=300).num_reels == 300
    with pytest.raises(ValueError):
        ReelsGenerateRequest(material_id="m1", num_reels=301)


def test_fresh_inventory_and_selector_cache_share_current_contract() -> None:
    assert {
        main.SELECTION_CONTRACT_VERSION,
        generation_jobs.REQUEST_SCHEMA_VERSION,
        ReelService.RANKED_FEED_CACHE_CONTRACT_VERSION,
    } == {"quality_silence_v27"}
    assert segment_cache.SELECTION_CONTRACT_VERSION == "quality_silence_v27"
    assert "quality_silence_v18" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v19" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v20" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v21" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v22" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v23" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v24" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v25" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v26" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v27" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS


def test_reel_response_schema_retains_v3_source_and_selector_metadata() -> None:
    serialized = ReelOut.model_validate({
        "reel_id": "schema-v3-reel",
        "material_id": "m1",
        "concept_id": "c1",
        "concept_title": "Cell energy",
        "video_url": "https://www.youtube.com/embed/AbCdEf12345?start=1&end=2",
        "t_start": 1.25,
        "t_end": 2.75,
        "transcript_snippet": "Cells transfer usable chemical energy through ATP.",
        "takeaways": [],
        "score": 0.91,
        "relevance_score": 0.13,
        "topic_relevance": 0.93,
        "selection_contract_version": "quality_silence_v4",
    }).model_dump()

    assert serialized["video_id"] == "AbCdEf12345"
    assert serialized["selection_contract_version"] == "quality_silence_v4"
    assert serialized["relevance_score"] == 0.13
    assert serialized["topic_relevance"] == 0.93


def test_public_generation_reel_keeps_legacy_relevance_semantics() -> None:
    public = main._public_generation_reel({
        "reel_id": "legacy-reel",
        "video_url": "https://www.youtube.com/embed/AbCdEf12345",
        "selection_contract_version": "quality_silence_v4",
        "relevance_score": 0.13,
        "_selection_topic_relevance": 0.93,
    })

    assert public["relevance_score"] == 0.13
    assert public["topic_relevance"] == 0.93
    assert not any(key.startswith("_selection_") for key in public)


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
        "transcript_snippet, takeaways_json, base_score, generation_id, "
        "search_context_json, created_at) "
        "VALUES (?, 'm1', 'c1', ?, '', 0, 30, "
        "'Cells use adenosine triphosphate to transfer usable chemical energy.', "
        "'[]', 1, ?, ?, ?)",
        (
            reel_id,
            video_id,
            generation_id,
            json.dumps({
                "surface_eligible": True,
                "boundary_status": "verified",
                "selection_contract_version": "quality_silence_v27",
                "speech_corridor_verified": True,
                "directly_teaches_topic": True,
                "substantive": True,
                "factually_grounded": True,
                "self_contained": True,
                "is_standalone": True,
                "informativeness": 0.9,
                "topic_relevance": 0.9,
                "educational_importance": 0.9,
                "quality_floor": 0.9,
                "quality_mean": 0.9,
                "topic_evidence_quote": (
                    "Cells use adenosine triphosphate to transfer usable chemical energy."
                ),
                "boundary_diagnostics": {
                    "method": "energy_silence",
                    "acoustic_verified": True,
                    "final_range": [0.0, 30.0],
                    "acoustic": {
                        "threshold_dbfs": -38.0,
                        "start_quiet": [0.0, 0.1],
                        "end_quiet": [29.9, 30.1],
                    },
                },
            }),
            created_at,
        ),
    )
    return {
        "reel_id": reel_id,
        "video_id": video_id,
        "t_start": 0.0,
        "t_end": 30.0,
        "selection_contract_version": "quality_silence_v27",
    }


def _set_reel_boundary_state(
    conn: sqlite3.Connection,
    *,
    reel_id: str,
    boundary_status: str,
    surface_eligible: bool = True,
    acoustic_verified: bool | None = None,
) -> None:
    row = conn.execute(
        "SELECT t_start, t_end FROM reels WHERE id = ?", (reel_id,)
    ).fetchone()
    assert row is not None
    start, end = float(row[0]), float(row[1])
    verified = (
        boundary_status == "verified"
        if acoustic_verified is None
        else acoustic_verified
    )
    if boundary_status == "context_aligned":
        boundary_diagnostics = {
            "method": "transcript_context",
            "context_aligned": True,
            "acoustic_verified": False,
            "transcript": {
                "context_aligned": True,
                "stage": "transcript",
                "reason": "complete_discourse_boundary",
                "required_speech_range": [start, end],
                "semantic_range": [start, end],
                "final_range": [start, end],
            },
        }
        caption_cues = [{
            "cue_id": "cue-1",
            "start": start,
            "end": end,
            "text": "Cells transfer usable chemical energy through ATP.",
        }]
    else:
        boundary_diagnostics = {
            "method": "energy_silence",
            "acoustic_verified": verified,
            "final_range": [start, end],
            "acoustic": (
                {
                    "threshold_dbfs": -38.0,
                    "start_quiet": [max(0.0, start - 0.1), start + 0.1],
                    "end_quiet": [max(0.0, end - 0.1), end + 0.1],
                }
                if verified
                else {}
            ),
        }
        caption_cues = []
    conn.execute(
        "UPDATE reels SET search_context_json = ? WHERE id = ?",
        (
            json.dumps({
                "surface_eligible": surface_eligible,
                "boundary_status": boundary_status,
                "speech_corridor_verified": True,
                "selection_caption_cues": caption_cues,
                "boundary_diagnostics": boundary_diagnostics,
                "selection_contract_version": "quality_silence_v27",
                "directly_teaches_topic": True,
                "substantive": True,
                "factually_grounded": True,
                "self_contained": True,
                "is_standalone": True,
                "informativeness": 0.9,
                "topic_relevance": 0.9,
                "educational_importance": 0.9,
                "quality_floor": 0.9,
                "quality_mean": 0.9,
                "topic_evidence_quote": (
                    "Cells use adenosine triphosphate to transfer usable chemical energy."
                ),
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
    content_fingerprint: str = "prior-fingerprint",
    request_params: dict | None = None,
) -> dict:
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id=concept_id,
        request_key=request_key,
        content_fingerprint=content_fingerprint,
        learner_id=learner_id,
        request_params=request_params or {"generation_mode": "slow", "num_reels": 12},
    )
    conn.execute(
        "UPDATE reel_generation_jobs SET status = ?, phase = 'terminal', "
        "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
        "WHERE id = ?",
        (status, generation_id, completed_at, completed_at, job["id"]),
    )
    return {**job, "status": status, "result_generation_id": generation_id}


def test_generation_job_reels_promote_internal_current_metadata_and_source(
    monkeypatch,
) -> None:
    conn = _conn()
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [{
            "reel_id": "live-final-reel",
            "material_id": "m1",
            "concept_id": "c1",
            "concept_title": "Cell energy",
            "video_url": (
                "https://www.youtube.com/embed/AbCdEf12345?start=1&end=2"
            ),
            "t_start": 1.25,
            "t_end": 2.75,
            "transcript_snippet": "Cells transfer usable chemical energy through ATP.",
            "takeaways": [],
            "score": 0.93,
            "relevance_score": 0.13,
            "_selection_contract_version": "quality_silence_v27",
            "_selection_topic_relevance": 0.93,
            "_selection_source_rank": 0,
        }],
    )
    try:
        reels = main._generation_job_reels(
            conn,
            {
                "result_generation_id": "live-generation",
                "material_id": "m1",
                "concept_id": "c1",
                "learner_id": "learner-1",
                "request_params_json": json.dumps({
                    "generation_mode": "slow",
                    "num_reels": 12,
                }),
            },
        )

        assert len(reels) == 1
        assert reels[0]["video_id"] == "AbCdEf12345"
        assert reels[0]["selection_contract_version"] == "quality_silence_v27"
        assert reels[0]["relevance_score"] == 0.93
        assert reels[0]["topic_relevance"] == 0.93
        assert not any(key.startswith("_selection_") for key in reels[0])
    finally:
        conn.close()


@pytest.mark.parametrize(
    "stale_version",
    [
        "quality_silence_v6",
        "quality_silence_v8",
        "quality_silence_v9",
        "quality_silence_v19",
        "quality_silence_v20",
        "quality_silence_v21",
        "quality_silence_v22",
        "quality_silence_v23",
        "quality_silence_v24",
        "quality_silence_v25",
        "quality_silence_v26",
    ],
)
def test_generation_job_reels_reject_stale_inventory(
    monkeypatch,
    stale_version: str,
) -> None:
    conn = _conn()
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [{
            "reel_id": "stale-reel",
            "selection_contract_version": stale_version,
        }],
    )
    try:
        assert main._generation_job_reels(
            conn,
            {
                "result_generation_id": "stale-generation",
                "material_id": "m1",
                "concept_id": "c1",
                "learner_id": "learner-1",
                "request_params_json": json.dumps({
                    "generation_mode": "slow",
                    "num_reels": 12,
                }),
            },
        ) == []
    finally:
        conn.close()


@pytest.mark.parametrize(
    "field",
    ["informativeness", "topic_relevance", "educational_importance"],
)
def test_reusable_generation_requires_every_quality_score_at_threshold(
    field: str,
) -> None:
    conn = _conn()
    generation_id = "quality-gate-generation"
    created_at = "2026-07-10T00:00:00+00:00"
    conn.execute(
        "INSERT INTO reel_generations "
        "(id, material_id, concept_id, request_key, generation_mode, "
        "retrieval_profile, status, reel_count, created_at) "
        "VALUES (?, 'm1', 'c1', 'quality-gate', 'fast', 'deep', "
        "'completed', 1, ?)",
        (generation_id, created_at),
    )
    _insert_generation_reel(
        conn,
        generation_id=generation_id,
        reel_id="quality-gate-reel",
        video_id="quality-gate-video",
        created_at=created_at,
    )
    try:
        row = conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'quality-gate-reel'"
        ).fetchone()
        context = json.loads(str(row["search_context_json"]))
        context[field] = 0.74
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'quality-gate-reel'",
            (json.dumps(context),),
        )
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is False

        context[field] = 0.75
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'quality-gate-reel'",
            (json.dumps(context),),
        )
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is True
    finally:
        conn.close()


def test_transcript_aligned_inventory_counts_reuses_and_replays_only_current_surface_rows() -> None:
    conn = _conn()
    generation_id = "transcript-boundary-generation"
    created_at = "2026-07-10T00:00:00+00:00"
    conn.execute(
        "INSERT INTO reel_generations "
        "(id, material_id, concept_id, request_key, generation_mode, "
        "retrieval_profile, status, reel_count, created_at) "
        "VALUES (?, 'm1', 'c1', 'transcript-boundary', 'fast', 'deep', "
        "'completed', 1, ?)",
        (generation_id, created_at),
    )
    _insert_generation_reel(
        conn,
        generation_id=generation_id,
        reel_id="transcript-boundary-reel",
        video_id="transcript-boundary-video",
        created_at=created_at,
    )
    try:
        _set_reel_boundary_state(
            conn,
            reel_id="transcript-boundary-reel",
            boundary_status="context_aligned",
        )
        assert main._count_generation_reels(conn, generation_id) == 1
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is True
        assert main._usable_boundary_reel_ids(
            conn, ["transcript-boundary-reel"]
        ) == {"transcript-boundary-reel"}

        context = json.loads(conn.execute(
            "SELECT search_context_json FROM reels WHERE id = ?",
            ("transcript-boundary-reel",),
        ).fetchone()[0])
        context["surface_eligible"] = False
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (json.dumps(context), "transcript-boundary-reel"),
        )
        assert main._count_generation_reels(conn, generation_id) == 0
        assert main._usable_boundary_reel_ids(
            conn, ["transcript-boundary-reel"]
        ) == set()

        context["surface_reason"] = "level_mismatch"
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (json.dumps(context), "transcript-boundary-reel"),
        )
        assert main._count_generation_reels(conn, generation_id) == 0
        assert main._usable_boundary_reel_ids(
            conn, ["transcript-boundary-reel"]
        ) == {"transcript-boundary-reel"}

        context["selection_contract_version"] = "quality_silence_v15"
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (json.dumps(context), "transcript-boundary-reel"),
        )
        assert main._count_generation_reels(conn, generation_id) == 0
        assert main._usable_boundary_reel_ids(
            conn, ["transcript-boundary-reel"]
        ) == set()
    finally:
        conn.close()


def test_generation_worker_uses_one_hobby_safe_process_worker(
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
            entered.set()
        assert release.wait(timeout=2.0)

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "lease_next_job", lease_next)
    monkeypatch.setattr(main, "_run_leased_generation_job", run_job)

    try:
        main._start_generation_worker()
        assert entered.wait(timeout=2.0)
        assert len(main._generation_worker_threads) == main.GENERATION_WORKER_COUNT == 1
        worker_threads = list(main._generation_worker_threads)
        main._start_generation_worker()
        assert main._generation_worker_threads == worker_threads
        assert len({owner for _job_id, owner in running_jobs}) == 1
        assert all(owner.startswith("worker-") for _job_id, owner in running_jobs)
        assert 0 < main.GENERATION_HEARTBEAT_SEC < main.GENERATION_LEASE_SEC
        assert main.GENERATION_WORKER_POLL_SEC > 600
        health = main.admin_health()
        assert health["generation_worker_alive"] is True
        assert health["generation_worker_count"] == 1
        assert health["generation_workers_alive"] == 1
        assert health["generation_worker_recovery_sec"] > 600
    finally:
        release.set()
        main._stop_generation_worker()
    assert main._generation_worker_threads == []


def test_idle_generation_worker_sweeps_once_then_wakes_immediately_on_signal(
    monkeypatch,
) -> None:
    main._stop_generation_worker()
    first_poll = threading.Event()
    second_poll = threading.Event()
    poll_count = 0
    poll_lock = threading.Lock()

    @contextmanager
    def connection(**_kwargs):
        yield object()

    def lease_next(_conn, **_kwargs):
        nonlocal poll_count
        with poll_lock:
            poll_count += 1
            if poll_count == 1:
                first_poll.set()
            elif poll_count == 2:
                second_poll.set()
        return None

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "lease_next_job", lease_next)

    try:
        main._start_generation_worker()
        assert first_poll.wait(timeout=1.0)
        time.sleep(0.05)
        with poll_lock:
            assert poll_count == 1

        main._wake_generation_worker()
        assert second_poll.wait(timeout=1.0)

        started = time.monotonic()
        main._stop_generation_worker()
        assert time.monotonic() - started < 1.0
    finally:
        main._stop_generation_worker()

    assert main._generation_worker_threads == []


def test_generation_worker_does_not_lose_wake_committed_during_sweep(
    monkeypatch,
) -> None:
    stop = threading.Event()
    poll_count = 0

    @contextmanager
    def connection(**_kwargs):
        yield object()

    def lease_next(_conn, **_kwargs):
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            main._wake_generation_worker()
        else:
            stop.set()
        return None

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "lease_next_job", lease_next)
    main._generation_worker_wake.clear()
    worker = threading.Thread(
        target=main._generation_worker_loop,
        args=("worker-race-test", stop),
    )
    worker.start()
    worker.join(timeout=1.0)
    if worker.is_alive():
        stop.set()
        main._wake_generation_worker()
        worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert poll_count == 2


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
        [StateThread(False)],
    )

    admin = main.admin_health()
    assert admin["ok"] is False
    assert admin["generation_worker_alive"] is False
    assert admin["generation_worker_count"] == 1
    assert admin["generation_workers_alive"] == 0
    assert admin["generation_worker_recovery_sec"] > 600
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
            "transcript_snippet, takeaways_json, base_score, generation_id, "
            "search_context_json, created_at) "
            "VALUES ('source-chain-reel', 'm1', 'c1', 'source-chain-video', '', "
            "0, 30, '', '[]', 1, ?, ?, ?)",
            (
                str(kwargs["generation_id"]),
                json.dumps({
                        "surface_eligible": True,
                        "boundary_status": "verified",
                        "selection_contract_version": "quality_silence_v27",
                        "speech_corridor_verified": True,
                        "directly_teaches_topic": True,
                        "substantive": True,
                        "factually_grounded": True,
                        "boundary_diagnostics": {
                            "acoustic_verified": True,
                            "final_range": [0.0, 30.0],
                                "acoustic": {
                                    "threshold_dbfs": -38.0,
                                    "start_quiet": [0.0, 0.1],
                                    "end_quiet": [29.9, 30.1],
                                },
                        },
                    }),
                now.isoformat(),
            ),
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", create_one_reel)
    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 0,
    )
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


def test_deferred_only_generation_remains_a_reusable_partial_reservoir(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    request_params = {
        "generation_mode": "slow",
        "num_reels": 1,
        "exclude_video_ids": [],
        "creative_commons_only": False,
        "min_relevance": None,
        "preferred_video_duration": "any",
        "knowledge_level": "beginner",
        "language": "en",
    }
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="deferred-only-request",
        content_fingerprint="same-fingerprint",
        learner_id="learner-1",
        request_params=request_params,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-deferred-only",
        now=now,
    )
    assert leased

    def generate_deferred_reel(worker_conn, **kwargs) -> None:
        _insert_generation_reel(
            worker_conn,
            generation_id=str(kwargs["generation_id"]),
            reel_id="deferred-only-reel",
            video_id="deferred-only-video",
            created_at=now.isoformat(),
        )
        row = worker_conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'deferred-only-reel'"
        ).fetchone()
        context = json.loads(str(row["search_context_json"]))
        context.update({
            "surface_eligible": False,
            "surface_reason": "level_mismatch",
            "deferred_level": True,
        })
        worker_conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'deferred-only-reel'",
            (json.dumps(context),),
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_deferred_reel)
    monkeypatch.setattr(main, "_generation_job_reels", lambda *_args, **_kwargs: [])
    try:
        main._run_leased_generation_job(leased, threading.Event())

        terminal_job = generation_jobs.get_job(conn, job["id"])
        assert terminal_job is not None
        generation_id = str(terminal_job["result_generation_id"] or "")
        assert terminal_job["status"] == "partial"
        assert generation_id
        assert terminal_job["terminal_error_code"] is None
        assert main._count_generation_reels(conn, generation_id) == 0

        generation = conn.execute(
            "SELECT status, reel_count, error_text FROM reel_generations WHERE id = ?",
            (generation_id,),
        ).fetchone()
        assert tuple(generation) == ("active", 0, None)
        head = conn.execute(
            "SELECT active_generation_id FROM reel_generation_heads "
            "WHERE material_id = 'm1' AND request_key = 'deferred-only-request'"
        ).fetchone()
        assert head["active_generation_id"] == generation_id

        events = generation_jobs.replay_events(conn, job_id=job["id"])
        final_event = next(event for event in events if event["type"] == "final")
        terminal_event = next(event for event in events if event["type"] == "terminal")
        assert final_event["payload"] == {
            "reels": [],
            "generation_id": generation_id,
            "authoritative": True,
        }
        assert terminal_event["payload"]["status"] == "partial"
        assert terminal_event["payload"]["result_generation_id"] == generation_id

        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="advanced-level-request",
            concept_id="c1",
            content_fingerprint="same-fingerprint",
            request_params={**request_params, "knowledge_level": "advanced"},
        ) == generation_id
    finally:
        conn.close()


def test_generation_worker_finalizes_a_rankable_nearest_level_fallback(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="rankable-nearest-level-fallback",
        content_fingerprint="same-fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 8,
            "knowledge_level": "beginner",
        },
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-rankable-fallback",
        now=now,
    )
    assert leased

    def generate_mixed_boundaries(worker_conn, **kwargs) -> None:
        generation_id = str(kwargs["generation_id"])
        _insert_generation_reel(
            worker_conn,
            generation_id=generation_id,
            reel_id="unusable-beginner-reel",
            video_id="unusable-beginner-video",
            created_at=now.isoformat(),
        )
        invalid_row = worker_conn.execute(
            "SELECT search_context_json FROM reels WHERE id = ?",
            ("unusable-beginner-reel",),
        ).fetchone()
        invalid_context = json.loads(str(invalid_row["search_context_json"]))
        invalid_context["boundary_status"] = "failed"
        worker_conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (json.dumps(invalid_context), "unusable-beginner-reel"),
        )

        _insert_generation_reel(
            worker_conn,
            generation_id=generation_id,
            reel_id="usable-intermediate-fallback",
            video_id="usable-intermediate-video",
            created_at=now.isoformat(),
        )
        fallback_row = worker_conn.execute(
            "SELECT search_context_json FROM reels WHERE id = ?",
            ("usable-intermediate-fallback",),
        ).fetchone()
        fallback_context = json.loads(str(fallback_row["search_context_json"]))
        fallback_context.update({
            "surface_eligible": False,
            "surface_reason": "level_mismatch",
            "deferred_level": True,
        })
        worker_conn.execute(
            "UPDATE reels SET difficulty = 0.5, search_context_json = ? WHERE id = ?",
            (
                json.dumps(fallback_context),
                "usable-intermediate-fallback",
            ),
        )

    monkeypatch.setattr(
        main.reel_service,
        "generate_reels",
        generate_mixed_boundaries,
    )
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [{
            "reel_id": "usable-intermediate-fallback",
            "difficulty": 0.5,
            "selection_contract_version": "quality_silence_v27",
        }],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        terminal_job = generation_jobs.get_job(conn, job["id"])
        assert terminal_job is not None
        assert terminal_job["status"] == "partial"
        assert terminal_job["terminal_error_code"] is None
        generation_id = str(terminal_job["result_generation_id"] or "")
        assert generation_id
        assert main._count_generation_reels(conn, generation_id) == 0
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is False

        events = generation_jobs.replay_events(conn, job_id=job["id"])
        final_event = next(event for event in events if event["type"] == "final")
        assert [
            reel["reel_id"] for reel in final_event["payload"]["reels"]
        ] == ["usable-intermediate-fallback"]
        assert final_event["payload"]["generation_id"] == generation_id
        terminal_event = next(
            event for event in events if event["type"] == "terminal"
        )
        assert terminal_event["payload"]["status"] == "partial"
    finally:
        conn.close()


def test_deferred_only_slow_reservoir_reuses_after_level_change_without_top_up(
    monkeypatch,
) -> None:
    conn = _conn()

    @contextmanager
    def connection(**_kwargs):
        yield conn

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(
        main, "_require_community_client_identity", lambda _request: "owner"
    )
    monkeypatch.setattr(
        main, "_resolve_learner_identity", lambda *_args, **_kwargs: "learner-1"
    )
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    provider = mock.Mock(side_effect=AssertionError("stored reservoir called provider"))
    monkeypatch.setattr(main.reel_service, "generate_reels", provider)
    monkeypatch.setattr(
        main.reel_service,
        "_score_text_relevance",
        lambda *_args, **_kwargs: {
            "score": 0.99,
            "concept_overlap": 1.0,
            "context_overlap": 1.0,
            "matched_terms": ["cell", "energy"],
            "off_topic_penalty": 0.0,
            "reason": "matched exact topic",
        },
    )
    monkeypatch.setattr(main.reel_service, "_build_caption_cues", lambda **_kwargs: [])

    now = datetime.now(timezone.utc)
    content_fingerprint = generation_jobs.material_content_fingerprint(conn, "m1", None)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="slow-beginner-deferred-only",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="advanced-deferred-reel",
        video_id="advanced-deferred-video",
        created_at=now.isoformat(),
    )
    row = conn.execute(
        "SELECT search_context_json FROM reels WHERE id = 'advanced-deferred-reel'"
    ).fetchone()
    context = json.loads(str(row["search_context_json"]))
    context.update({
        "surface_eligible": False,
        "surface_reason": "level_mismatch",
        "deferred_level": True,
        "boundary_confidence": 0.9,
        "selection_candidate_id": "advanced-deferred-video::advanced-unit",
    })
    conn.execute(
        "UPDATE reels SET difficulty = 0.85, search_context_json = ? "
        "WHERE id = 'advanced-deferred-reel'",
        (json.dumps(context),),
    )
    _terminal_job_for_generation(
        conn,
        request_key="slow-beginner-deferred-only",
        generation_id=source_generation_id,
        completed_at=now.isoformat(),
        concept_id=None,
        content_fingerprint=content_fingerprint,
        request_params={
            "generation_mode": "slow",
            "num_reels": 12,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": None,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "language": "en",
        },
    )
    main.reel_service.set_learner_level(conn, "m1", "learner-1", "advanced")
    initial_jobs = conn.execute(
        "SELECT COUNT(*) FROM reel_generation_jobs"
    ).fetchone()[0]
    initial_provider_rows = conn.execute(
        "SELECT COUNT(*) FROM generation_provider_usage"
    ).fetchone()[0]

    try:
        feed_response = main.feed(
            object(),
            material_id="m1",
            autofill=True,
            generation_mode="slow",
        )
        direct_response = asyncio.run(main.generate_reels(
            object(),
            ReelsGenerateRequest(
                material_id="m1",
                generation_mode="slow",
                num_reels=12,
            ),
        ))

        assert [reel["reel_id"] for reel in feed_response["reels"]] == [
            "advanced-deferred-reel"
        ]
        assert feed_response["generation_id"] == source_generation_id
        assert feed_response["generation_job_id"] is None
        assert [reel["reel_id"] for reel in direct_response["reels"]] == [
            "advanced-deferred-reel"
        ]
        assert direct_response["generation_id"] == source_generation_id
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == initial_jobs
        assert conn.execute(
            "SELECT COUNT(*) FROM generation_provider_usage"
        ).fetchone()[0] == initial_provider_rows
        provider.assert_not_called()
        wake.assert_not_called()
    finally:
        conn.close()


def test_cross_request_current_level_reservoir_does_not_force_fresh_top_up(
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
    calls: list[dict] = []

    def generate_stage(_worker_conn, **kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 5,
    )
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

        assert calls == []
        assert generation_jobs.get_job(conn, job["id"])["status"] == "completed"
    finally:
        conn.close()


def test_fast_source_generation_limits_slow_top_up_to_one_source(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="prior-fast-source-budget",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="prior-fast-current-level",
        video_id="prior-fast-video",
        created_at=now.isoformat(),
    )
    _terminal_job_for_generation(
        conn,
        request_key="prior-fast-source-budget",
        generation_id=source_generation_id,
        completed_at=now.isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "fast", "num_reels": 8},
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="slow-source-budget-delta",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "slow", "num_reels": 5},
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-source-budget-delta",
        now=now,
    )
    assert leased
    calls: list[dict] = []
    generated_count = 0

    def generate_stage(_worker_conn, **kwargs) -> None:
        nonlocal generated_count
        calls.append(kwargs)
        generated_count += int(kwargs["max_new_reels"])

    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(
        main,
        "_count_generation_reels",
        lambda _conn, generation_id: (
            0 if generation_id == source_generation_id else generated_count
        ),
    )
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
        assert calls[0]["retrieval_profile"] == "deep"
        assert calls[0]["max_generation_videos"] == 1
        assert calls[0]["max_new_reels"] == 4
        assert generation_jobs.get_job(conn, job["id"])["status"] == "completed"
    finally:
        conn.close()


def test_slow_generation_uses_one_deep_retrieval_with_three_source_cap(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="single-stage-source",
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
        request_key="single-stage-slow",
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
        lease_owner="worker-single-stage",
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
        kwargs["analyzed_video_ids"].update(
            {"deep-video-1", "deep-video-2", "deep-video-3"}
        )
        staged = [
            ("deep-reel-1", "deep-video-1"),
            ("deep-reel-2", "deep-video-2"),
            ("deep-reel-3", "deep-video-3"),
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
    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(main, "update_generation_progress", capture_progress)
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": "deep-reel-3"},
            {"reel_id": "source-reel"},
            {"reel_id": "deep-reel-1"},
            {"reel_id": "deep-reel-2"},
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert [call["retrieval_profile"] for call in calls] == ["deep"]
        assert [call["max_new_reels"] for call in calls] == [3]
        assert calls[0]["max_generation_videos"] == 3
        assert all(
            not call["generation_context"].require_acoustic_boundaries
            for call in calls
        )
        assert calls[0]["exclude_video_ids"] == ["manual-exclusion"]
        assert ("retrieval", 0.05) in progress_updates
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
        ] == ["deep-reel-1", "deep-reel-2", "deep-reel-3"]
        final_reel_ids = [
            reel["reel_id"]
            for event in events
            if event["type"] == "final"
            for reel in event["payload"]["reels"]
        ]
        assert final_reel_ids == [
            "deep-reel-3",
            "source-reel",
            "deep-reel-1",
            "deep-reel-2",
        ]
        assert "deep-reel-3" in final_reel_ids
        assert conn.execute(
            "SELECT COUNT(*) FROM reels WHERE generation_id = ?",
            (str((generation_jobs.get_job(conn, job["id"]) or {})["result_generation_id"]),),
        ).fetchone()[0] == 3
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("mode", "expected_reel_cap", "expected_source_cap"),
    [("fast", 8, 2), ("slow", 12, 3)],
)
def test_generation_mode_uses_one_deep_stage_and_mode_caps(
    monkeypatch,
    mode: str,
    expected_reel_cap: int,
    expected_source_cap: int,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"single-stage-cap-{mode}",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": mode, "num_reels": 300},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner=f"worker-{mode}-stage",
        now=now,
    )
    assert leased
    calls: list[dict] = []
    generated_count = 0

    def generate_stage(_worker_conn, **kwargs) -> None:
        nonlocal generated_count
        calls.append(kwargs)
        kwargs["analyzed_video_ids"].update(
            f"{mode}-video-{index}"
            for index in range(expected_source_cap)
        )
        for _source_index in range(expected_source_cap):
            kwargs["generation_context"].reserve_gemini_call(
                operation="flash_boundary_selector",
                model="gemini-3.5-flash",
                prompt_text="whole timestamped transcript",
                max_output_tokens=4096,
            )
        for index in range(kwargs["max_new_reels"]):
            reel = {
                "reel_id": f"{mode}-reel-{index}",
                "video_id": f"{mode}-video-{index % expected_source_cap}",
                "t_start": float(index * 10),
                "t_end": float(index * 10 + 8),
                "selection_contract_version": "quality_silence_v27",
            }
            kwargs["on_reel_created"](reel)
        generated_count += int(kwargs["max_new_reels"])

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_count_generation_reels",
        lambda *_args, **_kwargs: generated_count,
    )
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": f"{mode}-reel-{index}",
                "selection_contract_version": "quality_silence_v27",
            }
            for index in range(expected_reel_cap)
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert len(calls) == 1
        assert calls[0]["retrieval_profile"] == "deep"
        assert calls[0]["num_reels"] == expected_reel_cap
        assert calls[0]["max_generation_videos"] == expected_source_cap
        assert calls[0]["max_new_reels"] == expected_reel_cap
        gemini_budget = calls[0]["generation_context"].budget.snapshot()["gemini"]
        assert gemini_budget["flash_selector_calls"] == expected_source_cap
        assert gemini_budget["flash_selector_limit"] == expected_source_cap
        assert gemini_budget["pro_calls"] == 0
        assert gemini_budget["pro_call_limit"] == 0
        events = generation_jobs.replay_events(conn, job_id=job["id"])
        assert [event["type"] for event in events].count("candidate") == expected_reel_cap
        assert [event["type"] for event in events][-2:] == ["final", "terminal"]
        final_event = next(event for event in events if event["type"] == "final")
        terminal_event = next(event for event in events if event["type"] == "terminal")
        assert len(final_event["payload"]["reels"]) == expected_reel_cap
        assert terminal_event["payload"]["status"] == "completed"
    finally:
        conn.close()


def test_generation_worker_caps_stream_but_keeps_every_persisted_candidate(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="stream-all-persisted-candidates",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 8},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-stream-all",
        now=now,
    )
    assert leased

    def generate_many(worker_conn, **kwargs) -> None:
        for index in range(18):
            reel = _insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id=f"streamed-reel-{index}",
                video_id=f"streamed-video-{index}",
                created_at=now.isoformat(),
            )
            kwargs["on_reel_created"](reel)

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_many)
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": f"streamed-reel-{index}"}
            for index in range(8)
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        events = generation_jobs.replay_events(conn, job_id=job["id"])
        candidates = [
            event["payload"]["reel"]["reel_id"]
            for event in events
            if event["type"] == "candidate"
        ]
        assert candidates == [f"streamed-reel-{index}" for index in range(8)]
        completed_job = generation_jobs.get_job(conn, job["id"])
        assert completed_job is not None
        assert conn.execute(
            "SELECT COUNT(*) FROM reels WHERE generation_id = ?",
            (str(completed_job["result_generation_id"] or ""),),
        ).fetchone()[0] == 18
        assert completed_job["status"] == "completed"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "mode",
    ["slow", "fast"],
)
def test_single_stage_deadline_provider_failure_is_fatal(
    monkeypatch,
    mode: str,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"single-stage-deadline-{mode}",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": mode, "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner=f"worker-single-stage-deadline-{mode}",
        now=now,
    )
    assert leased
    calls: list[dict] = []

    def generate_stage(worker_conn, **kwargs) -> None:
        del worker_conn
        calls.append(kwargs)
        kwargs["analyzed_video_ids"].add("deep-analyzed-timeout")
        kwargs["retrieved_video_ids"].add("deep-retrieved-timeout")
        raise ProviderTransientError(
            "Supadata search timed out.",
            provider="supadata",
            operation="search",
            detail="generation deadline exceeded",
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed_job = generation_jobs.get_job(conn, job["id"])
        assert completed_job and completed_job["status"] == "failed"
        assert [call["retrieval_profile"] for call in calls] == ["deep"]
        assert calls[0]["max_generation_videos"] == (2 if mode == "fast" else 3)
        terminal_error_code = conn.execute(
            "SELECT terminal_error_code FROM reel_generation_jobs WHERE id = ?",
            (job["id"],),
        ).fetchone()[0]
        assert terminal_error_code == "provider_transient"
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
def test_single_stage_non_deadline_provider_failures_remain_fatal(
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
        request_key=f"single-stage-fatal-{provider_error.code}-{provider_error.detail}",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "slow", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-single-stage-fatal",
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
        assert profiles == ["deep"]
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
    inside_transaction = False

    @contextmanager
    def committed_connection(**_kwargs):
        nonlocal inside_transaction
        assert not inside_transaction
        inside_transaction = True
        try:
            yield conn
        finally:
            inside_transaction = False

    def assert_committed_wake() -> None:
        assert not inside_transaction

    wake = mock.Mock(side_effect=assert_committed_wake)
    monkeypatch.setattr(main, "get_conn", committed_connection)
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
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
        assert wake.call_count == 2
    finally:
        conn.close()


def test_generate_cost_guard_preserves_active_and_completed_reuse(monkeypatch) -> None:
    assert main.REELS_GENERATE_RATE_LIMIT_PER_WINDOW == 6
    assert main.GENERATION_GLOBAL_ACTIVE_LIMIT == 4
    assert main.GENERATION_LEARNER_ACTIVE_LIMIT == 1

    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    rate_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        main,
        "_enforce_rate_limit",
        lambda _request, scope, *, limit, **_kwargs: rate_calls.append((scope, limit)),
    )
    payload = ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=3)
    try:
        first = asyncio.run(main.generate_reels(object(), payload))
        first_job_id = json.loads(first.body)["job_id"]
        generation_jobs.submit_or_get_active(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key="other-active-request",
            content_fingerprint="other-fingerprint",
            learner_id="learner-1",
            request_params={},
        )

        coalesced = asyncio.run(main.generate_reels(object(), payload))
        assert json.loads(coalesced.body)["job_id"] == first_job_id

        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'completed', phase = 'terminal', "
            "progress = 1.0, completed_at = ?, updated_at = ? WHERE id = ?",
            (completed_at, completed_at, first_job_id),
        )
        generation_jobs.submit_or_get_active(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key="second-active-request",
            content_fingerprint="second-fingerprint",
            learner_id="learner-1",
            request_params={},
        )
        monkeypatch.setattr(
            main,
            "_generation_job_reels",
            lambda *_args, **_kwargs: [{"reel_id": "cached-reel"}],
        )

        cached = asyncio.run(main.generate_reels(object(), payload))
        assert cached["reels"] == [{"reel_id": "cached-reel"}]

        with pytest.raises(main.HTTPException) as captured:
            asyncio.run(main.generate_reels(
                object(),
                ReelsGenerateRequest(
                    material_id="m1",
                    concept_id="c1",
                    num_reels=3,
                    exclude_video_ids=["new-source"],
                ),
            ))
        assert captured.value.status_code == 429
        assert captured.value.detail == {
            "code": "generation_queue_full",
            "message": "Generation is busy. Retry after an active request finishes.",
            "scope": "learner",
            "limit": 1,
        }
        assert captured.value.headers == {"Retry-After": "30"}
        assert rate_calls == [("generation-submit", 6)]
    finally:
        conn.close()


def test_generate_reuses_partial_completed_result_without_quota_top_up(monkeypatch) -> None:
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

        assert top_up["reels"] == [{"reel_id": "only-existing-reel"}]
        assert top_up["generation_id"] == source_generation_id
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
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
    _insert_generation_reel(
        conn,
        generation_id="stream-generation",
        reel_id="provisional",
        video_id="stream-video",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _set_reel_boundary_state(
        conn,
        reel_id="provisional",
        boundary_status="context_aligned",
    )
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="candidate",
        payload={
            "reel": {
                "reel_id": "provisional",
                "selection_contract_version": "quality_silence_v27",
            },
            "provisional": True,
        },
        lease_owner="worker",
    )
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="candidate",
        payload={"reel": {"reel_id": "caption-only"}, "provisional": True},
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
        assert [event["seq"] for event in events] == [1, 3, 4]
        assert all(event["job_id"] == job["id"] and event["timestamp"] for event in events)
    finally:
        conn.close()


def test_authoritative_job_inventory_drops_candidates_absent_from_final_rank(monkeypatch) -> None:
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
                "selection_contract_version": "quality_silence_v27",
            },
            "provisional": True,
        },
        lease_owner="candidate-retention-worker",
        now=now,
    )
    _insert_generation_reel(
        conn,
        generation_id="generation-1",
        reel_id="streamed-reel",
        video_id="streamed-video",
        created_at=now.isoformat(),
    )
    _set_reel_boundary_state(
        conn,
        reel_id="streamed-reel",
        boundary_status="verified",
        acoustic_verified=False,
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [{
            "reel_id": "ranked-reel",
            "selection_contract_version": "quality_silence_v27",
        }],
    )
    monkeypatch.setattr(
        main,
        "_currently_surfaceable_generation_reel_ids",
        lambda *_args, **_kwargs: {"streamed-reel"},
    )
    job_row = {
        **leased,
        "result_generation_id": "generation-1",
    }

    stale_reels = main._generation_job_reels(conn, job_row)
    assert [reel["reel_id"] for reel in stale_reels] == ["ranked-reel"]

    _set_reel_boundary_state(
        conn,
        reel_id="streamed-reel",
        boundary_status="verified",
        acoustic_verified=True,
    )
    adaptive_context = json.loads(conn.execute(
        "SELECT search_context_json FROM reels WHERE id = 'streamed-reel'"
    ).fetchone()[0])
    adaptive_context["boundary_diagnostics"]["acoustic"] = {
        "adaptive_quiet": True,
        "threshold_dbfs": -38.0,
        "start_threshold_dbfs": -24.0,
        "end_threshold_dbfs": -38.0,
    }
    conn.execute(
        "UPDATE reels SET search_context_json = ? WHERE id = 'streamed-reel'",
        (json.dumps(adaptive_context),),
    )
    adaptive_reels = main._generation_job_reels(conn, job_row)
    assert [reel["reel_id"] for reel in adaptive_reels] == ["ranked-reel"]

    adaptive_context["boundary_diagnostics"]["acoustic"] = {
        "adaptive_quiet": False,
        "threshold_dbfs": -38.0,
        "start_threshold_dbfs": -38.0,
        "end_threshold_dbfs": -38.0,
    }
    conn.execute(
        "UPDATE reels SET search_context_json = ? WHERE id = 'streamed-reel'",
        (json.dumps(adaptive_context),),
    )
    reels = main._generation_job_reels(conn, job_row)

    assert [reel["reel_id"] for reel in reels] == ["ranked-reel"]

    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": f"ranked-reel-{index}",
                "video_id": f"ranked-video-{index}",
                "t_start": float(index * 30),
                "t_end": float(index * 30 + 20),
                "selection_contract_version": "quality_silence_v27",
            }
            for index in range(4)
        ],
    )
    capped = main._generation_job_reels(conn, job_row)
    assert len(capped) == 4
    assert {reel["reel_id"] for reel in capped} == {
        f"ranked-reel-{index}" for index in range(4)
    }
    conn.close()


def test_completed_job_status_and_replay_drop_currently_invalid_candidate(monkeypatch) -> None:
    conn = _conn()
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="completed-semantic-revalidation",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 2},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="semantic-revalidation-worker",
        now=now,
    )
    assert leased
    valid = _insert_generation_reel(
        conn,
        generation_id="semantic-generation",
        reel_id="current-valid",
        video_id="current-valid-video",
        created_at=now.isoformat(),
    )
    _set_reel_boundary_state(
        conn,
        reel_id="current-valid",
        boundary_status="context_aligned",
    )
    stale = _insert_generation_reel(
        conn,
        generation_id="semantic-generation",
        reel_id="historical-invalid",
        video_id="historical-invalid-video",
        created_at=(now + timedelta(seconds=1)).isoformat(),
    )
    for reel in (stale, valid):
        generation_jobs.append_event(
            conn,
            job_id=job["id"],
            lease_owner="semantic-revalidation-worker",
            event_type="candidate",
            payload={"reel": reel, "provisional": True},
            now=now,
        )
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        lease_owner="semantic-revalidation-worker",
        event_type="final",
        payload={"reels": [stale, valid], "authoritative": True},
        now=now,
    )
    terminal = generation_jobs.transition_terminal(
        conn,
        job_id=job["id"],
        lease_owner="semantic-revalidation-worker",
        status="completed",
        result_generation_id="semantic-generation",
        now=now,
    )
    assert terminal

    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [valid],
    )
    monkeypatch.setattr(
        main,
        "_currently_surfaceable_generation_reel_ids",
        lambda *_args, **_kwargs: {"current-valid"},
    )

    status = main._generation_job_status_payload(conn, terminal)
    replay = main._sanitize_generation_replay_events(
        conn,
        terminal,
        generation_jobs.replay_events(conn, job_id=job["id"]),
    )

    assert [reel["reel_id"] for reel in status["reels"]] == ["current-valid"]
    assert [
        event["payload"]["reel"]["reel_id"]
        for event in replay
        if event["type"] == "candidate"
    ] == ["current-valid"]
    final = next(event for event in replay if event["type"] == "final")
    assert [reel["reel_id"] for reel in final["payload"]["reels"]] == [
        "current-valid"
    ]
    conn.close()


@pytest.mark.parametrize(
    "fallback_reason",
    [
        "acoustic_refinement_unsafe",
        "start_silence_not_found",
        "audio_refinement_deadline_exceeded",
    ],
)
def test_boundary_only_failure_survives_persistence_feed_and_authoritative_replay(
    monkeypatch,
    fallback_reason: str,
) -> None:
    conn = _conn()

    @contextmanager
    def shared_connection(**_kwargs):
        yield conn

    monkeypatch.setattr(ingestion_pipeline, "get_conn", shared_connection)
    monkeypatch.setattr(
        main.reel_service,
        "learner_progress",
        lambda *_args, **_kwargs: {
            "selected_level": "beginner",
            "global_adjustment": 0.0,
            "feedback_revision": 0,
        },
    )

    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"boundary-fallback-{fallback_reason}",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    transcript_boundary = ingestion_pipeline._transcript_aligned_result(
        clip_engine_silence.SilenceVerificationResult(
            "unavailable",
            10.0,
            20.0,
            {"stage": "transcript", "reason": fallback_reason},
        ),
        speech_bounds=(10.0, 20.0),
        search_limits=(9.0, 21.0),
        projection_diagnostics={},
    )
    assert transcript_boundary.status == "context_aligned"

    transcript_diagnostics = dict(transcript_boundary.diagnostics)
    search_context = {
        "selection_candidate_id": f"candidate-{fallback_reason}",
        "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
        "surface_eligible": True,
        "speech_corridor_verified": True,
        "boundary_confidence": 0.9,
        "boundary_status": "context_aligned",
        "selection_caption_cues": [
            {
                "cue_id": "cue-1",
                "start": 10.0,
                "end": 20.0,
                "text": "Mitochondria produce ATP that cells use as chemical energy.",
            }
        ],
        "boundary_diagnostics": {
            "method": "transcript_context",
            "context_aligned": True,
            "acoustic_verified": False,
            "final_range": [10.0, 20.0],
            "transcript": transcript_diagnostics,
        },
        "directly_teaches_topic": True,
        "substantive": True,
        "factually_grounded": True,
        "self_contained": True,
        "is_standalone": True,
        "informativeness": 0.95,
        "topic_relevance": 0.95,
        "educational_importance": 0.95,
        "quality_floor": 0.95,
        "quality_mean": 0.95,
        "topic_evidence_quote": (
            "Mitochondria produce ATP that cells use as chemical energy."
        ),
    }
    pipeline = ingestion_pipeline.IngestionPipeline(
        youtube_service=None,
        embedding_service=None,
        serverless_mode=False,
    )
    persisted = pipeline._persist_ingest(
        adapter_result=YouTubeSourceRef(
            source_id="BoundaryVid",
            source_url="https://www.youtube.com/watch?v=BoundaryVid",
            playback_url="https://www.youtube.com/embed/BoundaryVid",
        ),
        metadata=IngestMetadata(
            platform="yt",
            source_id="BoundaryVid",
            source_url="https://www.youtube.com/watch?v=BoundaryVid",
            playback_url="https://www.youtube.com/embed/BoundaryVid",
            title="Mitochondria and cellular energy",
            duration_sec=30.0,
        ),
        cues=[
            IngestTranscriptCue(
                cue_id="cue-1",
                start=10.0,
                end=20.0,
                text="Mitochondria produce ATP that cells use as chemical energy.",
            )
        ],
        chosen=IngestSegment(
            t_start=10.0,
            t_end=20.0,
            text="Mitochondria produce ATP that cells use as chemical energy.",
            score=0.95,
        ),
        snippet="Mitochondria produce ATP that cells use as chemical energy.",
        material_id="m1",
        concept_id="c1",
        clip_window=(10.0, 20.0),
        target_max=0,
        generation_id=generation_id,
        clip_title="How mitochondria supply cellular energy",
        clip_difficulty=0.2,
        clip_details={
            "cue_ids": ["cue-1"],
            "informativeness": 0.95,
            "search_context": search_context,
        },
    )

    ranked_feed = main.reel_service.ranked_feed(
        conn,
        "m1",
        fast_mode=True,
        generation_id=generation_id,
        learner_id="learner-1",
        require_verified_boundaries=True,
    )
    assert [reel["reel_id"] for reel in ranked_feed] == [persisted.reel_id]

    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"job-{fallback_reason}",
        content_fingerprint="boundary-fallback-fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 8},
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="boundary-fallback-worker",
    )
    assert leased
    candidate_reel = {
        "reel_id": persisted.reel_id,
        "video_id": "BoundaryVid",
        "t_start": persisted.t_start,
        "t_end": persisted.t_end,
        "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
    }
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="candidate",
        payload={"reel": candidate_reel, "provisional": True},
        lease_owner="boundary-fallback-worker",
    )
    generation_jobs.append_event(
        conn,
        job_id=job["id"],
        event_type="final",
        payload={"reels": [candidate_reel], "authoritative": True},
        lease_owner="boundary-fallback-worker",
    )
    terminal = generation_jobs.transition_terminal(
        conn,
        job_id=job["id"],
        status="completed",
        lease_owner="boundary-fallback-worker",
        result_generation_id=generation_id,
    )
    assert terminal

    status = main._generation_job_status_payload(conn, terminal)
    replay = main._sanitize_generation_replay_events(
        conn,
        terminal,
        generation_jobs.replay_events(conn, job_id=job["id"]),
    )
    assert [reel["reel_id"] for reel in status["reels"]] == [persisted.reel_id]
    assert [
        event["payload"]["reel"]["reel_id"]
        for event in replay
        if event["type"] == "candidate"
    ] == [persisted.reel_id]
    final = next(event for event in replay if event["type"] == "final")
    assert final["payload"]["authoritative"] is True
    assert [reel["reel_id"] for reel in final["payload"]["reels"]] == [
        persisted.reel_id
    ]
    stored_context = json.loads(conn.execute(
        "SELECT search_context_json FROM reels WHERE id = ?",
        (persisted.reel_id,),
    ).fetchone()[0])
    assert stored_context["boundary_status"] == "context_aligned"
    assert stored_context["boundary_diagnostics"]["transcript"]["reason"] == (
        fallback_reason
    )
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
    rate_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        main,
        "_enforce_rate_limit",
        lambda _request, scope, *, limit, **_kwargs: rate_calls.append((scope, limit)),
    )
    inside_transaction = False

    @contextmanager
    def committed_connection(**_kwargs):
        nonlocal inside_transaction
        assert not inside_transaction
        inside_transaction = True
        try:
            yield conn
        finally:
            inside_transaction = False

    def assert_committed_wake() -> None:
        assert not inside_transaction

    wake = mock.Mock(side_effect=assert_committed_wake)
    monkeypatch.setattr(main, "get_conn", committed_connection)
    monkeypatch.setattr(main, "_wake_generation_worker", wake)

    def fail_on_unversioned_feed(*_args, **_kwargs):
        raise AssertionError("fresh feed must not rank base/legacy inventory")

    monkeypatch.setattr(main, "_ranked_request_reels", fail_on_unversioned_feed)
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
        wake.assert_called_once_with()
        assert rate_calls == [
            ("feed", 36),
            ("feed", 36),
            ("generation-submit", 6),
        ]
    finally:
        conn.close()


@pytest.mark.parametrize("status", ["completed", "partial", "exhausted"])
def test_feed_reload_reports_the_successful_terminal_job(
    monkeypatch,
    status: str,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    terminal = {
        "id": f"terminal-{status}",
        "status": status,
        "result_generation_id": f"generation-{status}",
    }
    monkeypatch.setattr(
        main,
        "find_completed_generation_job",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        main,
        "find_active_generation_job",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(main, "_ranked_request_reels", lambda *_args, **_kwargs: [])
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    try:
        response = main.feed(object(), material_id="m1", autofill=True)

        assert response["generation_job_id"] == terminal["id"]
        assert response["generation_job_status"] == status
        wake.assert_not_called()
    finally:
        conn.close()


@pytest.mark.parametrize("throttle", ["capacity", "rate"])
def test_feed_returns_ranked_reels_when_autofill_is_throttled(
    monkeypatch,
    throttle: str,
) -> None:
    assert main.FEED_RATE_LIMIT_PER_WINDOW == 36
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(
        main,
        "_fetch_active_generation_row",
        lambda *_args, **_kwargs: {"id": "existing-generation"},
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": f"ranked-{index}"} for index in range(5)
        ],
    )
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    try:
        if throttle == "capacity":
            for index in range(main.GENERATION_LEARNER_ACTIVE_LIMIT):
                generation_jobs.submit_or_get_active(
                    conn,
                    material_id="m1",
                    concept_id=None,
                    request_key=f"other-feed-request-{index}",
                    content_fingerprint=f"other-fingerprint-{index}",
                    learner_id="learner-1",
                    request_params={},
                )
        else:
            def reject_generation_submit(_request, scope, **_kwargs):
                if scope == "generation-submit":
                    raise main.HTTPException(status_code=429, detail="Too many requests.")

            monkeypatch.setattr(
                main,
                "_enforce_rate_limit",
                reject_generation_submit,
            )

        response = main.feed(object(), material_id="m1", autofill=True)

        assert [reel["reel_id"] for reel in response["reels"]] == [
            f"ranked-{index}" for index in range(5)
        ]
        assert response["generation_job_id"] is None
        assert response["generation_job_status"] is None
        wake.assert_not_called()
    finally:
        conn.close()


def test_feed_slow_reservoir_immediately_satisfies_fast_without_queuing(
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
    for index in range(3):
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"verified-prior-reel-{index}",
            video_id=f"verified-prior-video-{index}",
            created_at=completed_at,
        )
    _terminal_job_for_generation(
        conn,
        request_key="prior-level-request",
        generation_id=source_generation_id,
        completed_at=completed_at,
        concept_id=None,
        content_fingerprint=generation_jobs.material_content_fingerprint(
            conn, "m1", None
        ),
        request_params={
            "generation_mode": "slow",
            "num_reels": 12,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": None,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "language": "en",
        },
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
        response = main.feed(
            object(),
            material_id="m1",
            autofill=True,
            generation_mode="fast",
        )

        assert response["generation_id"] == source_generation_id
        assert len(response["reels"]) == 5
        assert ranked_generation_ids == [source_generation_id]
        queued = conn.execute(
            "SELECT * FROM reel_generation_jobs WHERE status = 'queued' "
            "ORDER BY created_at DESC, id DESC LIMIT 1"
        ).fetchone()
        assert queued is None
    finally:
        conn.close()


def test_feed_fast_reservoir_queues_slow_top_up_despite_full_ranked_page(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    completed_at = datetime.now(timezone.utc).isoformat()
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="prior-fast-feed-request",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    for index in range(2):
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"prior-fast-feed-reel-{index}",
            video_id=f"prior-fast-feed-video-{index}",
            created_at=completed_at,
        )
    _terminal_job_for_generation(
        conn,
        request_key="prior-fast-feed-request",
        generation_id=source_generation_id,
        completed_at=completed_at,
        concept_id=None,
        content_fingerprint=generation_jobs.material_content_fingerprint(
            conn, "m1", None
        ),
        request_params={
            "generation_mode": "fast",
            "num_reels": 8,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": None,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "language": "en",
        },
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": f"prior-fast-feed-reel-{index}"}
            for index in range(12)
        ],
    )
    try:
        response = main.feed(
            object(),
            material_id="m1",
            autofill=True,
            generation_mode="slow",
        )

        assert len(response["reels"]) == 5
        assert response["generation_id"] == source_generation_id
        assert response["generation_job_id"]
        queued = generation_jobs.get_job(conn, response["generation_job_id"])
        assert queued is not None
        assert queued["status"] == "queued"
        assert queued["source_generation_id"] == source_generation_id
    finally:
        conn.close()


def test_feed_queues_only_when_cross_level_reservoir_has_no_current_level_clip(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="future-level-feed-reservoir",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    monkeypatch.setattr(
        main,
        "_verified_cross_request_source_generation",
        lambda *_args, **_kwargs: source_generation_id,
    )
    monkeypatch.setattr(main, "_ranked_request_reels", lambda *_args, **_kwargs: [])
    try:
        response = main.feed(object(), material_id="m1", autofill=True)

        assert response["reels"] == []
        assert response["generation_job_id"]
        queued = generation_jobs.get_job(conn, response["generation_job_id"])
        assert queued is not None
        assert queued["source_generation_id"] == source_generation_id
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("concept_id", "expected_clause", "expected_params"),
    [
        (
            None,
            "AND concept_id IS NULL",
            ("m1", "learner-1", "new-request", "current-fingerprint"),
        ),
        (
            "c1",
            "AND concept_id = ?",
            ("m1", "learner-1", "new-request", "current-fingerprint", "c1"),
        ),
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
        return []

    monkeypatch.setattr(main, "fetch_all", record_candidate_query)

    result = main._verified_cross_request_source_generation(
        object(),
        material_id="m1",
        learner_id="learner-1",
        request_key="new-request",
        concept_id=concept_id,
        content_fingerprint="current-fingerprint",
        request_params={
            "creative_commons_only": False,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "language": "en",
            "exclude_video_ids": [],
            "min_relevance": None,
        },
    )

    assert result is None
    assert expected_clause in str(observed["sql"])
    assert "? IS NULL" not in str(observed["sql"])
    assert observed["params"] == expected_params


@pytest.mark.parametrize(
    ("current_fingerprint", "request_override", "compatible"),
    [
        ("changed-fingerprint", {}, False),
        ("same-fingerprint", {"creative_commons_only": True}, False),
        ("same-fingerprint", {"preferred_video_duration": "long"}, False),
        ("same-fingerprint", {"knowledge_level": "advanced"}, True),
        ("same-fingerprint", {"language": "fr"}, False),
        ("same-fingerprint", {"exclude_video_ids": ["prior-video"]}, False),
        ("same-fingerprint", {"min_relevance": 0.9}, False),
    ],
)
def test_cross_request_reuse_checks_source_constraints_but_reuses_all_difficulties(
    current_fingerprint: str,
    request_override: dict,
    compatible: bool,
) -> None:
    conn = _conn()
    completed_at = datetime.now(timezone.utc).isoformat()
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="prior-compatible-shape",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=generation_id,
        reel_id="prior-reel",
        video_id="prior-video",
        created_at=completed_at,
    )
    prior_params = {
        "generation_mode": "fast",
        "num_reels": 8,
        "exclude_video_ids": [],
        "creative_commons_only": False,
        "min_relevance": None,
        "preferred_video_duration": "any",
        "knowledge_level": "beginner",
        "language": "en",
    }
    _terminal_job_for_generation(
        conn,
        request_key="prior-compatible-shape",
        generation_id=generation_id,
        completed_at=completed_at,
        content_fingerprint="same-fingerprint",
        request_params=prior_params,
    )
    current_params = {**prior_params, "generation_mode": "slow", **request_override}
    try:
        result = main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="new-request",
            concept_id="c1",
            content_fingerprint=current_fingerprint,
            request_params=current_params,
        )
        assert result == (generation_id if compatible else None)
    finally:
        conn.close()


def test_cross_request_reuse_skips_newer_invalid_chain_for_older_valid_chain() -> None:
    conn = _conn()
    completed_at = datetime.now(timezone.utc)
    request_params = {
        "generation_mode": "fast",
        "num_reels": 8,
        "exclude_video_ids": [],
        "creative_commons_only": False,
        "min_relevance": None,
        "preferred_video_duration": "any",
        "knowledge_level": "beginner",
        "language": "en",
    }
    older_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="older-valid-request",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=older_generation_id,
        reel_id="older-valid-reel",
        video_id="older-valid-video",
        created_at=(completed_at - timedelta(seconds=1)).isoformat(),
    )
    _terminal_job_for_generation(
        conn,
        request_key="older-valid-request",
        generation_id=older_generation_id,
        completed_at=(completed_at - timedelta(seconds=1)).isoformat(),
        content_fingerprint="same-fingerprint",
        request_params=request_params,
    )

    newer_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="newer-invalid-request",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=newer_generation_id,
        reel_id="newer-invalid-reel",
        video_id="newer-invalid-video",
        created_at=completed_at.isoformat(),
    )
    _set_reel_boundary_state(
        conn,
        reel_id="newer-invalid-reel",
        boundary_status="verified",
        acoustic_verified=False,
    )
    _terminal_job_for_generation(
        conn,
        request_key="newer-invalid-request",
        generation_id=newer_generation_id,
        completed_at=completed_at.isoformat(),
        content_fingerprint="same-fingerprint",
        request_params=request_params,
    )
    try:
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="current-request",
            concept_id="c1",
            content_fingerprint="same-fingerprint",
            request_params={**request_params, "generation_mode": "slow"},
        ) == older_generation_id
    finally:
        conn.close()


def test_v7_feed_merges_value_ranked_batches_without_breaking_batch_topology(
    monkeypatch,
) -> None:
    conn = _conn()
    root_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="ordered-root",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    child_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="ordered-child",
        generation_mode="slow",
        retrieval_profile="unified",
        source_generation_id=root_generation_id,
    )

    def reel(
        reel_id: str,
        *,
        difficulty: float,
        relevance: float,
        quality: float,
        source_rank: int,
        start: float,
    ) -> dict:
        return {
            "reel_id": reel_id,
            "video_id": reel_id,
            "t_start": start,
            "t_end": start + 10.0,
            "difficulty": difficulty,
            "_selection_quality_floor": quality,
            "_selection_quality_mean": quality,
            "_selection_topic_relevance": relevance,
            "_selection_source_rank": source_rank,
            "_selection_ordered": True,
            "selection_contract_version": "quality_silence_v27",
        }

    root_reels = [
        reel(
            "easy-stage",
            difficulty=0.1,
            relevance=0.2,
            quality=0.80,
            source_rank=0,
            start=5.0,
        ),
        reel(
            "root-prerequisite",
            difficulty=0.1,
            relevance=0.7,
            quality=0.79,
            source_rank=0,
            start=20.0,
        ),
        reel(
            "root-dependent-high-value",
            difficulty=0.1,
            relevance=0.99,
            quality=0.99,
            source_rank=0,
            start=30.0,
        ),
    ]
    child_reels = [
        reel(
            "child-best-independent",
            difficulty=0.1,
            relevance=0.80,
            quality=0.95,
            source_rank=1,
            start=90.0,
        ),
        reel(
            "child-mean-independent",
            difficulty=0.1,
            relevance=0.95,
            quality=0.85,
            source_rank=0,
            start=1.0,
        ),
        reel(
            "child-low-value",
            difficulty=0.1,
            relevance=0.99,
            quality=0.10,
            source_rank=0,
            start=1.0,
        ),
    ]

    monkeypatch.setattr(
        main.reel_service,
        "ranked_feed",
        lambda *_args, **kwargs: (
            [dict(item) for item in root_reels]
            if kwargs.get("generation_id") == root_generation_id
            else [dict(item) for item in child_reels]
        ),
    )
    monkeypatch.setattr(
        main,
        "_shape_request_page_reels",
        lambda rows, **_kwargs: list(rows),
    )
    try:
        ranked = main._ranked_request_reels(
            conn,
            material_id="m1",
            fast_mode=False,
            generation_id=child_generation_id,
            min_relevance=None,
            preferred_video_duration="any",
            target_clip_duration_sec=0,
            target_clip_duration_min_sec=None,
            target_clip_duration_max_sec=None,
            page=1,
            limit=20,
        )

        assert [item["reel_id"] for item in ranked] == [
            "child-best-independent",
            "child-mean-independent",
            "easy-stage",
            "root-prerequisite",
            "root-dependent-high-value",
            "child-low-value",
        ]
    finally:
        conn.close()


def test_generation_chain_uses_nearest_difficulty_across_all_batches(
    monkeypatch,
) -> None:
    conn = _conn()
    root_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="difficulty-root",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    child_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="difficulty-child",
        generation_mode="slow",
        retrieval_profile="unified",
        source_generation_id=root_generation_id,
    )

    def reel(reel_id: str, difficulty: float) -> dict:
        return {
            "reel_id": reel_id,
            "video_id": reel_id,
            "t_start": 10.0,
            "t_end": 20.0,
            "difficulty": difficulty,
            "_selection_quality_floor": 0.9,
            "_selection_quality_mean": 0.9,
            "_selection_topic_relevance": 0.9,
            "_selection_source_rank": 0,
            "_selection_ordered": True,
            "selection_contract_version": "quality_silence_v27",
        }

    monkeypatch.setattr(
        main.reel_service,
        "ranked_feed",
        lambda *_args, **kwargs: (
            [reel("advanced-only", 0.85)]
            if kwargs.get("generation_id") == root_generation_id
            else [reel("intermediate-only", 0.50)]
        ),
    )
    monkeypatch.setattr(
        main,
        "_shape_request_page_reels",
        lambda rows, **_kwargs: list(rows),
    )
    try:
        ranked = main._ranked_request_reels(
            conn,
            material_id="m1",
            fast_mode=False,
            generation_id=child_generation_id,
            min_relevance=None,
            preferred_video_duration="any",
            target_clip_duration_sec=0,
            target_clip_duration_min_sec=None,
            target_clip_duration_max_sec=None,
            page=1,
            limit=20,
        )

        assert [item["reel_id"] for item in ranked] == ["intermediate-only"]
    finally:
        conn.close()


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
    conn.execute(
        "UPDATE reels SET search_context_json = '{}' WHERE id = 'legacy-child-reel'"
    )
    _terminal_job_for_generation(
        conn,
        request_key="prior-child",
        generation_id=child_generation_id,
        completed_at=completed_at,
        concept_id=None,
        status="partial",
        content_fingerprint="same-fingerprint",
        request_params={
            "generation_mode": "fast",
            "num_reels": 8,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": None,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "language": "en",
        },
    )
    try:
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="new-level-request",
            concept_id=None,
            content_fingerprint="same-fingerprint",
            request_params={
                "exclude_video_ids": [],
                "creative_commons_only": False,
                "min_relevance": None,
                "preferred_video_duration": "any",
                "knowledge_level": "beginner",
                "language": "en",
            },
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
            content_fingerprint="same-fingerprint",
            request_params={
                "exclude_video_ids": [],
                "creative_commons_only": False,
                "min_relevance": None,
                "preferred_video_duration": "any",
                "knowledge_level": "beginner",
                "language": "en",
            },
        ) is None

        _set_reel_boundary_state(
            conn,
            reel_id="legacy-child-reel",
            boundary_status="verified",
            acoustic_verified=False,
        )
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="new-level-request",
            concept_id=None,
            content_fingerprint="same-fingerprint",
            request_params={
                "exclude_video_ids": [],
                "creative_commons_only": False,
                "min_relevance": None,
                "preferred_video_duration": "any",
                "knowledge_level": "beginner",
                "language": "en",
            },
        ) is None
    finally:
        conn.close()


def test_generate_slow_reservoir_immediately_satisfies_fast_without_queuing(
    monkeypatch,
) -> None:
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
    for index in range(3):
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"verified-concept-reel-{index}",
            video_id=f"verified-concept-video-{index}",
            created_at=completed_at,
        )
    _terminal_job_for_generation(
        conn,
        request_key="prior-concept-level",
        generation_id=source_generation_id,
        completed_at=completed_at,
        content_fingerprint=generation_jobs.material_content_fingerprint(
            conn, "m1", "c1"
        ),
        request_params={
            "generation_mode": "slow",
            "num_reels": 12,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": None,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "language": "en",
        },
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": "verified-concept-reel",
                "video_id": "verified-concept-video-0",
                "selection_contract_version": "quality_silence_v27",
            }
        ],
    )
    monkeypatch.setattr(
        main,
        "submit_generation_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compatible current-level reservoir queued generation")
        ),
    )
    try:
        response = asyncio.run(main.generate_reels(
            object(),
            ReelsGenerateRequest(
                material_id="m1",
                concept_id="c1",
                generation_mode="fast",
                num_reels=5,
            ),
        ))
        assert response["generation_id"] == source_generation_id
        assert response["reels"] == [
            {
                "reel_id": "verified-concept-reel",
                "video_id": "verified-concept-video-0",
                "selection_contract_version": "quality_silence_v27",
            }
        ]
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_generate_fast_reservoir_queues_slow_top_up(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    completed_at = datetime.now(timezone.utc).isoformat()
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="prior-fast-concept-request",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    for index in range(2):
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"prior-fast-concept-reel-{index}",
            video_id=f"prior-fast-concept-video-{index}",
            created_at=completed_at,
        )
    _terminal_job_for_generation(
        conn,
        request_key="prior-fast-concept-request",
        generation_id=source_generation_id,
        completed_at=completed_at,
        content_fingerprint=generation_jobs.material_content_fingerprint(
            conn, "m1", "c1"
        ),
        request_params={
            "generation_mode": "fast",
            "num_reels": 8,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": None,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "language": "en",
        },
    )
    monkeypatch.setattr(
        main,
        "_reused_generation_reels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("insufficient Fast coverage reused as complete Slow inventory")
        ),
    )
    try:
        response = asyncio.run(main.generate_reels(
            object(),
            ReelsGenerateRequest(
                material_id="m1",
                concept_id="c1",
                generation_mode="slow",
                num_reels=12,
            ),
        ))

        assert isinstance(response, JSONResponse)
        assert response.status_code == 202
        queued = generation_jobs.get_job(conn, json.loads(response.body)["job_id"])
        assert queued is not None
        assert queued["status"] == "queued"
        assert queued["source_generation_id"] == source_generation_id
    finally:
        conn.close()


def test_generate_queues_when_cross_level_reservoir_has_no_current_level_clip(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="future-level-reservoir",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    monkeypatch.setattr(
        main,
        "_verified_cross_request_source_generation",
        lambda *_args, **_kwargs: source_generation_id,
    )
    monkeypatch.setattr(main, "_reused_generation_reels", lambda *_args, **_kwargs: [])
    try:
        response = asyncio.run(main.generate_reels(
            object(),
            ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=5),
        ))
        assert isinstance(response, JSONResponse)
        assert response.status_code == 202
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
