from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from backend.app import db
from backend.app.config import get_settings
from backend.app.ingestion.persistence import ensure_clip_concept
from backend.app.services import generation_jobs as jobs


BASE_TIME = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(db.SCHEMA)
    db._migrate_durable_generation_foundation_sqlite(conn)
    conn.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-1", "Cell biology", "topic", BASE_TIME.isoformat()),
    )
    conn.execute(
        """
        INSERT INTO concepts (id, material_id, title, keywords_json, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "concept-1",
            "material-1",
            "Mitochondria",
            '["energy", "respiration"]',
            "How cells produce usable energy.",
            BASE_TIME.isoformat(),
        ),
    )
    conn.commit()
    return conn


def _submit(conn: sqlite3.Connection, *, request_key: str = "request-1", now=BASE_TIME):
    return jobs.submit_or_get_active(
        conn,
        material_id="material-1",
        concept_id="concept-1",
        request_key=request_key,
        content_fingerprint="fingerprint-1",
        learner_id="learner-1",
        request_params={"generation_mode": "slow"},
        now=now,
    )


def test_submitted_job_stamps_current_request_schema_without_mutating_input() -> None:
    conn = _memory_conn()
    params = {"generation_mode": "slow"}
    try:
        row, created = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="schema-stamp",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params=params,
            now=BASE_TIME,
        )

        assert created is True
        assert params == {"generation_mode": "slow"}
        assert row["max_attempts"] == 3
        assert json.loads(row["request_params_json"])["request_schema_version"] == (
            jobs.REQUEST_SCHEMA_VERSION
        )
    finally:
        conn.close()


def test_active_capacity_coalesces_identical_requests_before_rejecting_new_work() -> None:
    conn = _memory_conn()
    created_callbacks: list[str] = []
    try:
        first, created = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="capacity-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={},
            max_global_active_jobs=1,
            max_active_jobs_per_learner=1,
            before_create=lambda: created_callbacks.append("created"),
        )
        same, same_created = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="capacity-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={},
            max_global_active_jobs=1,
            max_active_jobs_per_learner=1,
            before_create=lambda: created_callbacks.append("coalesced"),
        )

        assert created is True
        assert same_created is False
        assert same["id"] == first["id"]
        with pytest.raises(jobs.GenerationQueueFullError) as captured:
            jobs.submit_or_get_active(
                conn,
                material_id="material-1",
                concept_id="concept-1",
                request_key="different-request",
                content_fingerprint="fingerprint-1",
                learner_id="learner-2",
                request_params={},
                max_global_active_jobs=1,
                max_active_jobs_per_learner=1,
                before_create=lambda: created_callbacks.append("rejected"),
            )
        assert captured.value.scope == "global"
        assert created_callbacks == ["created"]
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs WHERE status IN ('queued', 'running')"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_postgres_submission_lock_is_acquired_before_active_request_recheck(
    monkeypatch,
) -> None:
    events: list[str] = []
    winner = {"id": "concurrent-winner", "status": "queued"}

    def find_active(*_args, **_kwargs):
        events.append("active")
        return None if events.count("active") == 1 else winner

    def fetch_lock(_conn, query, params=()):
        assert "pg_advisory_xact_lock" in query
        assert params == (jobs.GENERATION_SUBMIT_ADVISORY_LOCK_ID,)
        events.append("lock")
        return {"acquired": None}

    monkeypatch.setattr(jobs, "find_active_job", find_active)
    monkeypatch.setattr(jobs, "fetch_one", fetch_lock)

    row, created = jobs.submit_or_get_active(
        object(),
        material_id="material-1",
        concept_id=None,
        request_key="concurrent-request",
        content_fingerprint="fingerprint-1",
        learner_id="learner-1",
        request_params={},
        max_global_active_jobs=4,
        max_active_jobs_per_learner=2,
        before_create=lambda: events.append("create"),
    )

    assert row is winner
    assert created is False
    assert events == ["active", "lock", "active"]


def test_active_capacity_limits_each_learner_without_consuming_remaining_global_slots() -> None:
    conn = _memory_conn()
    try:
        for index in range(1):
            jobs.submit_or_get_active(
                conn,
                material_id="material-1",
                concept_id="concept-1",
                request_key=f"learner-one-{index}",
                content_fingerprint="fingerprint-1",
                learner_id="learner-1",
                request_params={},
                max_global_active_jobs=4,
                max_active_jobs_per_learner=1,
            )

        with pytest.raises(jobs.GenerationQueueFullError) as captured:
            jobs.submit_or_get_active(
                conn,
                material_id="material-1",
                concept_id="concept-1",
                request_key="learner-one-overflow",
                content_fingerprint="fingerprint-1",
                learner_id="learner-1",
                request_params={},
                max_global_active_jobs=4,
                max_active_jobs_per_learner=1,
            )
        assert captured.value.scope == "learner"

        other, created = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="learner-two-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-2",
            request_params={},
            max_global_active_jobs=4,
            max_active_jobs_per_learner=1,
        )
        assert created is True
        assert other["learner_id"] == "learner-2"
    finally:
        conn.close()


def test_request_key_uses_content_and_truthful_controls() -> None:
    assert jobs.ALL_STATUSES == {
        "queued",
        "running",
        "completed",
        "partial",
        "exhausted",
        "failed",
        "cancelled",
    }
    conn = _memory_conn()
    try:
        fingerprint = jobs.material_content_fingerprint(conn, "material-1", "concept-1")
        base = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )
        explicit_empty_adaptation = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
            adaptation_fingerprint=jobs.EMPTY_ADAPTATION_FINGERPRINT,
        )
        assert explicit_empty_adaptation == base
        changed_level = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="advanced",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )
        assert base != changed_level

        changed_duration = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=999,
            target_clip_duration_min_sec=-10,
            target_clip_duration_max_sec=1,
        )
        assert changed_duration == base

        changed_relevance = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
            min_relevance=0.9,
        )
        assert changed_relevance != base

        excluded_source = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
            exclude_video_ids=["yt:video-b", "video-a"],
        )
        reordered_exclusions = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=999,
            target_clip_duration_min_sec=-10,
            target_clip_duration_max_sec=1,
            exclude_video_ids=["video-a", "video-b"],
        )
        assert excluded_source != base
        assert reordered_exclusions == excluded_source

        continued_batch = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
            continuation_token="job-batch-1",
        )
        repeated_continuation = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=999,
            target_clip_duration_min_sec=-10,
            target_clip_duration_max_sec=1,
            continuation_token=" job-batch-1 ",
        )
        next_batch = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
            continuation_token="job-batch-2",
        )
        assert continued_batch != base
        assert repeated_continuation == continued_batch
        assert next_batch != continued_batch

        other_learner = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-2",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )
        assert other_learner != base

        changed_adaptation = jobs.build_request_key(
            material_id="material-1",
            concept_id="concept-1",
            content_fingerprint=fingerprint,
            learner_id="learner-1",
            knowledge_level="beginner",
            generation_mode="slow",
            creative_commons_only=False,
            source_duration="medium",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
            adaptation_fingerprint="concept-c1:confusing",
        )
        assert changed_adaptation != base

        conn.execute("UPDATE concepts SET summary = 'Changed content' WHERE id = 'concept-1'")
        assert jobs.material_content_fingerprint(conn, "material-1", "concept-1") != fingerprint
    finally:
        conn.close()


def test_material_fingerprint_ignores_generated_clip_facets_but_tracks_source_concepts() -> None:
    conn = _memory_conn()
    try:
        initial = jobs.material_content_fingerprint(conn, "material-1")
        generated_id, _title, _key = ensure_clip_concept(
            conn,
            material_id="material-1",
            title="ATP synthesis",
        )
        family_generated_id, family_title, _key = ensure_clip_concept(
            conn,
            material_id="material-1",
            title="worked ATP example",
            semantic_identity="ATP synthesis",
        )
        legacy_ordinal_id, _title, _key = ensure_clip_concept(
            conn,
            material_id="material-1",
            title="Newton's 1st law",
        )

        assert jobs.material_content_fingerprint(conn, "material-1") == initial
        assert jobs.material_content_fingerprint(conn, "material-1", generated_id) != initial
        assert family_title == "ATP synthesis"
        assert jobs.material_content_fingerprint(
            conn, "material-1", family_generated_id
        ) != initial
        assert jobs.material_content_fingerprint(
            conn, "material-1", legacy_ordinal_id
        ) != initial

        conn.execute(
            """
            INSERT INTO concepts (id, material_id, title, keywords_json, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "source-concept-2",
                "material-1",
                "Cell membrane",
                '["membrane"]',
                "A source-derived concept.",
                BASE_TIME.isoformat(),
            ),
        )
        assert jobs.material_content_fingerprint(conn, "material-1") != initial
    finally:
        conn.close()


@pytest.mark.parametrize(
    "stale_version",
    [
        "adaptive_clip_concepts_v5",
        "adaptive_clip_concepts_v3",
        "adaptive_clip_concepts_v2",
        "quality_silence_v6",
        "quality_silence_v9",
        "quality_silence_v10",
        "quality_silence_v18",
        "quality_silence_v19",
        "quality_silence_v20",
        "quality_silence_v21",
        "quality_silence_v22",
        "quality_silence_v23",
        "quality_silence_v24",
        "quality_silence_v25",
        "quality_silence_v26",
        "quality_silence_v27",
        "quality_silence_v28",
        "quality_silence_v29",
        "quality_silence_v30",
        "quality_silence_v31",
        "quality_silence_v32",
        "quality_silence_v33",
    ],
)
def test_request_key_version_invalidates_stale_inventory(
    monkeypatch,
    stale_version: str,
) -> None:
    params = {
        "material_id": "material-1",
        "concept_id": "concept-1",
        "content_fingerprint": "fingerprint-1",
        "learner_id": "learner-1",
        "knowledge_level": "beginner",
        "generation_mode": "slow",
        "creative_commons_only": False,
        "source_duration": "medium",
        "target_clip_duration_sec": 55,
        "target_clip_duration_min_sec": 20,
        "target_clip_duration_max_sec": 55,
    }
    assert jobs.REQUEST_SCHEMA_VERSION == "adaptive_clip_concepts_v7"
    verified_key = jobs.build_request_key(**params)
    monkeypatch.setattr(jobs, "REQUEST_SCHEMA_VERSION", stale_version)

    assert jobs.build_request_key(**params) != verified_key


def test_sqlite_init_migrates_legacy_job_and_metadata_tables(tmp_path, monkeypatch) -> None:
    path = tmp_path / "studyreels.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE videos (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, channel_title TEXT, description TEXT,
            duration_sec INTEGER, view_count INTEGER DEFAULT 0,
            is_creative_commons INTEGER DEFAULT 0, provider TEXT DEFAULT 'youtube',
            playback_url TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE reels (
            id TEXT PRIMARY KEY, generation_id TEXT, material_id TEXT NOT NULL,
            concept_id TEXT NOT NULL, video_id TEXT NOT NULL, video_url TEXT NOT NULL,
            t_start REAL NOT NULL, t_end REAL NOT NULL, transcript_snippet TEXT NOT NULL,
            takeaways_json TEXT NOT NULL, ai_summary TEXT NOT NULL DEFAULT '',
            match_reason TEXT NOT NULL DEFAULT '', informativeness REAL,
            base_score REAL NOT NULL, difficulty REAL, created_at TEXT NOT NULL
        );
        CREATE TABLE reel_generation_jobs (
            id TEXT PRIMARY KEY, material_id TEXT NOT NULL, concept_id TEXT,
            request_key TEXT NOT NULL, source_generation_id TEXT NOT NULL,
            result_generation_id TEXT, target_profile TEXT NOT NULL DEFAULT 'deep',
            request_params_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'queued',
            created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, error_text TEXT
        );
        """
    )
    legacy.executemany(
        """
        INSERT INTO reel_generation_jobs
            (id, material_id, request_key, source_generation_id, status, created_at)
        VALUES (?, 'material-1', 'same-request', 'source-1', ?, ?)
        """,
        [
            ("running-job", "running", "2026-07-10T00:00:00+00:00"),
            ("queued-job", "queued", "2026-07-10T00:01:00+00:00"),
        ],
    )
    legacy.commit()
    legacy.close()

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    db._db_ready = False
    try:
        db.init_db()
        migrated = sqlite3.connect(path)
        migrated.row_factory = sqlite3.Row
        try:
            job_columns = {
                row[1] for row in migrated.execute("PRAGMA table_info(reel_generation_jobs)")
            }
            assert {
                "learner_id",
                "phase",
                "progress",
                "lease_owner",
                "lease_expires_at",
                "heartbeat_at",
                "retry_not_before_at",
                "attempt_count",
                "max_attempts",
                "deadline_at",
                "cancel_requested",
                "model_used",
                "quality_degraded",
                "usage_json",
                "terminal_error_code",
                "next_event_seq",
            }.issubset(job_columns)
            table_names = {
                row[0]
                for row in migrated.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            assert {
                "generation_job_events",
                "supadata_search_cache",
                "transcript_artifacts",
                "generation_provider_usage",
                "concept_search_terms",
                "blocked_video_tombstones",
            }.issubset(table_names)
            indexes = {
                row[1] for row in migrated.execute("PRAGMA index_list(reel_generation_jobs)")
            }
            assert "idx_reel_generation_jobs_active_request" in indexes
            migrated_jobs = {
                row["id"]: (row["status"], row["terminal_error_code"])
                for row in migrated.execute(
                    "SELECT id, status, terminal_error_code FROM reel_generation_jobs ORDER BY id"
                )
            }
            assert migrated_jobs == {
                "queued-job": ("cancelled", "legacy_job_contract"),
                "running-job": ("cancelled", "legacy_job_contract"),
            }
        finally:
            migrated.close()
    finally:
        db._db_ready = False
        get_settings.cache_clear()


def test_concurrent_submit_returns_one_active_job(tmp_path) -> None:
    path = tmp_path / "jobs.db"
    setup = sqlite3.connect(path)
    setup.row_factory = sqlite3.Row
    setup.executescript(db.SCHEMA)
    db._migrate_durable_generation_foundation_sqlite(setup)
    setup.commit()
    setup.close()

    barrier = threading.Barrier(2)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []

    def submit() -> None:
        conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            barrier.wait(timeout=5)
            row, created = jobs.submit_or_get_active(
                conn,
                material_id="material-1",
                concept_id=None,
                request_key="shared-request",
                content_fingerprint="fingerprint",
                learner_id="learner",
                request_params={},
                now=BASE_TIME,
            )
            results.append((str(row["id"]), created))
        except BaseException as exc:
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=submit), threading.Thread(target=submit)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert errors == []
    assert len(results) == 2
    assert len({job_id for job_id, _created in results}) == 1
    assert sorted(created for _job_id, created in results) == [False, True]
    check = sqlite3.connect(path)
    try:
        assert check.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs WHERE status IN ('queued', 'running')"
        ).fetchone()[0] == 1
    finally:
        check.close()


def test_lease_expiry_recovery_and_heartbeat() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn)
        first = jobs.lease_job(conn, job_id=job["id"], lease_owner="worker-a", now=BASE_TIME)
        assert first and first["status"] == "running" and first["attempt_count"] == 1

        assert jobs.heartbeat_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            now=BASE_TIME + timedelta(seconds=30),
        )
        assert jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-b",
            now=BASE_TIME + timedelta(seconds=100),
        ) is None

        recovered = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-b",
            now=BASE_TIME + timedelta(seconds=121),
        )
        assert recovered and recovered["lease_owner"] == "worker-b"
        assert recovered["attempt_count"] == 2
    finally:
        conn.close()


def test_retry_cap_fails_expired_third_attempt_with_terminal_event() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn)
        assert jobs.lease_job(conn, job_id=job["id"], lease_owner="worker-a", now=BASE_TIME)
        assert jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-b",
            now=BASE_TIME + timedelta(seconds=91),
        )
        assert jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-c",
            now=BASE_TIME + timedelta(seconds=182),
        )
        assert jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-d",
            now=BASE_TIME + timedelta(seconds=273),
        ) is None

        failed = jobs.get_job(conn, job["id"])
        assert failed and failed["status"] == "failed"
        assert failed["attempt_count"] == 3
        assert failed["terminal_error_code"] == "attempts_exhausted"
        events = jobs.replay_events(conn, job_id=job["id"])
        assert [(event["seq"], event["type"]) for event in events] == [(1, "terminal")]
    finally:
        conn.close()


def test_retryable_failure_requeues_same_job_without_terminalizing_or_settling(
    monkeypatch,
) -> None:
    conn = _memory_conn()
    settled: list[str] = []
    monkeypatch.setattr(jobs, "_settle_search_quota", lambda _conn, job_id: settled.append(job_id))
    try:
        job, _ = _submit(conn, request_key="retryable-requeue")
        first = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            now=BASE_TIME,
        )
        assert first and first["attempt_count"] == 1

        requeued = jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            expected_attempt_count=1,
            usage={"retry_errors": [{"code": "provider_transient"}]},
            now=BASE_TIME + timedelta(seconds=1),
        )
        assert requeued and requeued["status"] == "queued"
        assert requeued["phase"] == "retrying"
        assert requeued["attempt_count"] == 1
        assert requeued["result_generation_id"] == first["result_generation_id"]
        assert requeued["lease_owner"] is None
        assert json.loads(requeued["usage_json"])["retry_errors"][0]["code"] == (
            "provider_transient"
        )
        assert jobs.replay_events(conn, job_id=job["id"]) == []
        assert settled == []

        replayed = jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            expected_attempt_count=1,
            now=BASE_TIME + timedelta(seconds=1),
        )
        assert replayed and replayed["status"] == "queued"

        second = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-b",
            now=BASE_TIME + timedelta(seconds=2),
        )
        assert second and second["attempt_count"] == 2
        assert jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="stale-worker",
            expected_attempt_count=2,
            now=BASE_TIME + timedelta(seconds=3),
        ) is None
        assert jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="worker-b",
            expected_attempt_count=2,
            now=BASE_TIME + timedelta(seconds=3),
        )

        third = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-c",
            now=BASE_TIME + timedelta(seconds=4),
        )
        assert third and third["attempt_count"] == 3
        assert jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="worker-c",
            expected_attempt_count=3,
            now=BASE_TIME + timedelta(seconds=5),
        ) is None
        jobs.transition_terminal(
            conn,
            job_id=job["id"],
            status="failed",
            lease_owner="worker-c",
            error_code="provider_transient",
            now=BASE_TIME + timedelta(seconds=5),
        )
        assert [
            event["type"] for event in jobs.replay_events(conn, job_id=job["id"])
        ] == ["terminal"]
        assert settled == [job["id"]]
    finally:
        conn.close()


def test_retryable_failure_requeue_respects_deadline_and_cancellation(
    monkeypatch,
) -> None:
    conn = _memory_conn()
    settled: list[str] = []
    monkeypatch.setattr(jobs, "_settle_search_quota", lambda _conn, job_id: settled.append(job_id))
    try:
        deadline_job, _ = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="retry-deadline",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={},
            now=BASE_TIME,
            deadline_seconds=10,
        )
        deadline_lease = jobs.lease_job(
            conn,
            job_id=deadline_job["id"],
            lease_owner="deadline-worker",
            now=BASE_TIME,
        )
        assert deadline_lease
        assert jobs.requeue_retryable_failure(
            conn,
            job_id=deadline_job["id"],
            lease_owner="deadline-worker",
            expected_attempt_count=1,
            now=BASE_TIME + timedelta(seconds=10),
        ) is None
        assert jobs.replay_events(conn, job_id=deadline_job["id"]) == []
        assert jobs.lease_job(
            conn,
            job_id=deadline_job["id"],
            lease_owner="late-worker",
            now=BASE_TIME + timedelta(seconds=10),
        ) is None
        deadline_terminal = jobs.get_job(conn, deadline_job["id"])
        assert deadline_terminal and deadline_terminal["status"] == "failed"
        assert deadline_terminal["terminal_error_code"] == "deadline_exceeded"

        cancelled_job, _ = _submit(
            conn,
            request_key="retry-cancelled",
            now=BASE_TIME + timedelta(seconds=20),
        )
        cancelled_lease = jobs.lease_job(
            conn,
            job_id=cancelled_job["id"],
            lease_owner="cancelled-worker",
            now=BASE_TIME + timedelta(seconds=20),
        )
        assert cancelled_lease
        jobs.request_cancellation(
            conn,
            job_id=cancelled_job["id"],
            now=BASE_TIME + timedelta(seconds=21),
        )
        replayed_cancel = jobs.requeue_retryable_failure(
            conn,
            job_id=cancelled_job["id"],
            lease_owner="cancelled-worker",
            expected_attempt_count=1,
            now=BASE_TIME + timedelta(seconds=21),
        )
        assert replayed_cancel and replayed_cancel["status"] == "cancelled"
        assert [
            event["type"]
            for event in jobs.replay_events(conn, job_id=cancelled_job["id"])
        ] == ["terminal"]
        assert settled == [deadline_job["id"], cancelled_job["id"]]
    finally:
        conn.close()


def test_rate_limit_requeue_waits_until_retry_not_before() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn, request_key="retry-rate-limit")
        first = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="rate-limit-worker-1",
            now=BASE_TIME,
        )
        assert first
        requeued = jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="rate-limit-worker-1",
            expected_attempt_count=1,
            retry_after_sec=5.0,
            now=BASE_TIME + timedelta(seconds=1),
        )
        assert requeued and requeued["status"] == "queued"
        assert datetime.fromisoformat(requeued["retry_not_before_at"]) == (
            BASE_TIME + timedelta(seconds=6)
        )
        assert jobs.next_queued_retry_delay(
            conn,
            now=BASE_TIME + timedelta(seconds=2),
        ) == pytest.approx(4.0)
        assert jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="rate-limit-worker-early",
            now=BASE_TIME + timedelta(seconds=5),
        ) is None
        still_queued = jobs.get_job(conn, job["id"])
        assert still_queued and still_queued["status"] == "queued"
        second = jobs.lease_next_job(
            conn,
            lease_owner="rate-limit-worker-2",
            now=BASE_TIME + timedelta(seconds=6),
        )
        assert second and second["id"] == job["id"]
        assert second["attempt_count"] == 2
        assert second["retry_not_before_at"] is None
    finally:
        conn.close()


def test_rate_limit_without_provider_hint_uses_bounded_attempt_backoff() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn, request_key="retry-rate-limit-default")
        first = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="rate-limit-default-worker-1",
            now=BASE_TIME,
        )
        assert first
        usage = {
            "retry_errors": [
                {"code": "provider_rate_limited", "retry_after_sec": None}
            ]
        }
        requeued = jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="rate-limit-default-worker-1",
            expected_attempt_count=1,
            usage=usage,
            now=BASE_TIME + timedelta(seconds=1),
        )
        assert requeued
        assert datetime.fromisoformat(requeued["retry_not_before_at"]) == (
            BASE_TIME + timedelta(seconds=6)
        )
        second = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="rate-limit-default-worker-2",
            now=BASE_TIME + timedelta(seconds=6),
        )
        assert second and second["attempt_count"] == 2
        second_requeue = jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="rate-limit-default-worker-2",
            expected_attempt_count=2,
            usage=usage,
            now=BASE_TIME + timedelta(seconds=7),
        )
        assert second_requeue
        assert datetime.fromisoformat(
            second_requeue["retry_not_before_at"]
        ) == BASE_TIME + timedelta(seconds=22)
    finally:
        conn.close()


def test_non_rate_limit_without_provider_hint_requeues_immediately() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn, request_key="retry-transient-no-delay")
        first = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="transient-default-worker",
            now=BASE_TIME,
        )
        assert first
        requeued = jobs.requeue_retryable_failure(
            conn,
            job_id=job["id"],
            lease_owner="transient-default-worker",
            expected_attempt_count=1,
            usage={"retry_errors": [{"code": "provider_transient"}]},
            now=BASE_TIME + timedelta(seconds=1),
        )
        assert requeued and requeued["retry_not_before_at"] is None
    finally:
        conn.close()


def test_queued_first_attempt_gets_fresh_execution_deadline_when_leased() -> None:
    conn = _memory_conn()
    try:
        job, _ = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="deadline-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={},
            now=BASE_TIME,
            deadline_seconds=10,
        )
        original_submission_window = datetime.fromisoformat(job["deadline_at"])
        assert original_submission_window == BASE_TIME + timedelta(seconds=10)

        claimed_at = BASE_TIME + timedelta(seconds=jobs.DEFAULT_QUEUE_TTL_SECONDS - 1)
        leased = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="late-worker",
            now=claimed_at,
        )
        assert leased and leased["status"] == "running"
        assert leased["lease_owner"] == "late-worker"
        assert leased["attempt_count"] == 1
        assert datetime.fromisoformat(leased["started_at"]) == claimed_at
        assert datetime.fromisoformat(leased["deadline_at"]) == claimed_at + timedelta(
            seconds=10
        )
        assert datetime.fromisoformat(leased["lease_expires_at"]) == claimed_at + timedelta(
            seconds=10
        )
    finally:
        conn.close()


def test_durable_queued_first_attempt_survives_queue_ttl_and_gets_fresh_deadline() -> None:
    conn = _memory_conn()
    try:
        job, _ = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="durable-queue-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={jobs.DURABLE_QUEUE_WAIT_PARAM: True},
            now=BASE_TIME,
            deadline_seconds=10,
        )
        claimed_at = BASE_TIME + timedelta(
            seconds=jobs.DEFAULT_QUEUE_TTL_SECONDS + 60
        )

        assert jobs.expire_stale_queued_job(
            conn,
            job_id=job["id"],
            now=claimed_at,
        ) is False
        leased = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="durable-queue-worker",
            now=claimed_at,
        )

        assert leased and leased["status"] == "running"
        assert leased["attempt_count"] == 1
        assert datetime.fromisoformat(leased["started_at"]) == claimed_at
        assert datetime.fromisoformat(leased["deadline_at"]) == (
            claimed_at + timedelta(seconds=10)
        )
    finally:
        conn.close()


def test_stale_queued_first_attempt_times_out_once_and_cannot_be_leased() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn, request_key="stale-queue-request")
        expired_at = BASE_TIME + timedelta(seconds=jobs.DEFAULT_QUEUE_TTL_SECONDS)

        assert jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            now=expired_at,
        ) is None
        expired = jobs.get_job(conn, job["id"])
        assert expired and expired["status"] == "failed"
        assert expired["attempt_count"] == 0
        assert expired["terminal_error_code"] == "queue_timeout"
        assert expired["lease_owner"] is None

        jobs.expire_stale_queued_job(conn, job_id=job["id"], now=expired_at)
        assert jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-b",
            now=expired_at + timedelta(seconds=1),
        ) is None
        events = jobs.replay_events(conn, job_id=job["id"])
        assert [(event["seq"], event["type"]) for event in events] == [(1, "terminal")]
        assert events[0]["payload"]["error"]["code"] == "queue_timeout"
    finally:
        conn.close()


def test_concurrent_queue_expiration_emits_one_terminal_event(tmp_path) -> None:
    path = tmp_path / "queue-expiration.db"
    setup = sqlite3.connect(path, isolation_level=None)
    setup.row_factory = sqlite3.Row
    setup.executescript(db.SCHEMA)
    db._migrate_durable_generation_foundation_sqlite(setup)
    setup.execute(
        "INSERT INTO materials (id, raw_text, source_type, created_at) VALUES (?, ?, ?, ?)",
        ("material-1", "Cell biology", "topic", BASE_TIME.isoformat()),
    )
    setup.execute(
        "INSERT INTO concepts (id, material_id, title, keywords_json, summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("concept-1", "material-1", "Mitochondria", "[]", "Cell energy", BASE_TIME.isoformat()),
    )
    job, _ = _submit(setup, request_key="concurrent-queue-expiration")
    setup.close()

    barrier = threading.Barrier(2)
    results: list[bool] = []
    errors: list[BaseException] = []
    expired_at = BASE_TIME + timedelta(seconds=jobs.DEFAULT_QUEUE_TTL_SECONDS)

    def expire() -> None:
        conn = sqlite3.connect(path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            barrier.wait(timeout=2.0)
            results.append(
                jobs.expire_stale_queued_job(conn, job_id=job["id"], now=expired_at)
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=expire) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    verify = sqlite3.connect(path)
    verify.row_factory = sqlite3.Row
    try:
        assert errors == []
        assert sorted(results) == [False, True]
        assert jobs.get_job(verify, job["id"])["terminal_error_code"] == "queue_timeout"
        assert [event["type"] for event in jobs.replay_events(verify, job_id=job["id"])] == [
            "terminal"
        ]
    finally:
        verify.close()


def test_submit_expires_stale_active_job_before_creating_replacement() -> None:
    conn = _memory_conn()
    try:
        stale, created = _submit(conn, request_key="replace-stale-request")
        assert created is True

        replacement, replacement_created = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="replace-stale-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={"generation_mode": "slow"},
            now=BASE_TIME + timedelta(seconds=jobs.DEFAULT_QUEUE_TTL_SECONDS),
        )

        assert replacement_created is True
        assert replacement["id"] != stale["id"]
        assert replacement["status"] == "queued"
        expired = jobs.get_job(conn, stale["id"])
        assert expired and expired["status"] == "failed"
        assert expired["terminal_error_code"] == "queue_timeout"
        assert [
            event["type"] for event in jobs.replay_events(conn, job_id=stale["id"])
        ] == ["terminal"]
    finally:
        conn.close()


def test_deadline_and_lease_windows_are_bounded_by_contract() -> None:
    conn = _memory_conn()
    try:
        assert jobs.DEFAULT_HEARTBEAT_SECONDS == 15
        job, _ = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="bounded-window-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={},
            now=BASE_TIME,
            deadline_seconds=10_000,
        )
        assert datetime.fromisoformat(job["deadline_at"]) == BASE_TIME + timedelta(
            seconds=jobs.DEFAULT_DEADLINE_SECONDS
        )

        leased = jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            now=BASE_TIME,
            lease_seconds=10_000,
        )
        assert leased
        assert datetime.fromisoformat(leased["lease_expires_at"]) == BASE_TIME + timedelta(
            seconds=90
        )
    finally:
        conn.close()


def test_deadline_clamps_heartbeats_and_rejects_stale_worker_writes() -> None:
    conn = _memory_conn()
    try:
        job, _ = jobs.submit_or_get_active(
            conn,
            material_id="material-1",
            concept_id="concept-1",
            request_key="hard-deadline-request",
            content_fingerprint="fingerprint-1",
            learner_id="learner-1",
            request_params={},
            now=BASE_TIME,
            deadline_seconds=10,
        )
        leased = jobs.lease_job(
            conn, job_id=job["id"], lease_owner="worker-a", now=BASE_TIME
        )
        assert leased
        assert datetime.fromisoformat(leased["lease_expires_at"]) == BASE_TIME + timedelta(
            seconds=10
        )
        assert jobs.heartbeat_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            now=BASE_TIME + timedelta(seconds=5),
        )
        refreshed = jobs.get_job(conn, job["id"])
        assert refreshed
        assert datetime.fromisoformat(refreshed["lease_expires_at"]) == BASE_TIME + timedelta(
            seconds=10
        )

        deadline = BASE_TIME + timedelta(seconds=10)
        assert not jobs.update_progress(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            phase="segmenting",
            progress=0.5,
            now=deadline,
        )
        with pytest.raises(jobs.JobLeaseLostError):
            jobs.append_event(
                conn,
                job_id=job["id"],
                lease_owner="worker-a",
                event_type="candidate",
                payload={"reel_id": "too-late"},
                now=deadline,
            )
        with pytest.raises(jobs.JobLeaseLostError):
            jobs.transition_terminal(
                conn,
                job_id=job["id"],
                lease_owner="worker-a",
                status="completed",
                now=deadline,
            )

        assert jobs.lease_next_job(conn, lease_owner="worker-b", now=deadline) is None
        failed = jobs.get_job(conn, job["id"])
        assert failed and failed["status"] == "failed"
        assert failed["terminal_error_code"] == "deadline_exceeded"
        assert [event["type"] for event in jobs.replay_events(conn, job_id=job["id"])] == [
            "terminal"
        ]
    finally:
        conn.close()


def test_append_event_rolls_back_sequence_when_event_insert_fails() -> None:
    conn = _memory_conn()
    try:
        conn.isolation_level = None
        job, _ = _submit(conn, request_key="event-rollback-request")
        conn.execute(
            """
            INSERT INTO generation_job_events
                (job_id, seq, event_type, payload_json, created_at)
            VALUES (?, 1, 'candidate', '{}', ?)
            """,
            (job["id"], BASE_TIME.isoformat()),
        )

        with pytest.raises(db.DatabaseIntegrityError):
            jobs.append_event(
                conn,
                job_id=job["id"],
                event_type="candidate",
                payload={"reel_id": "duplicate-sequence"},
                now=BASE_TIME + timedelta(seconds=1),
            )

        stored = jobs.get_job(conn, job["id"])
        assert stored and stored["next_event_seq"] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM generation_job_events WHERE job_id = ?",
            (job["id"],),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_cancellation_is_idempotent_and_revokes_running_lease() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn)
        assert jobs.lease_job(conn, job_id=job["id"], lease_owner="worker-a", now=BASE_TIME)
        cancelled = jobs.request_cancellation(
            conn, job_id=job["id"], now=BASE_TIME + timedelta(seconds=5)
        )
        assert cancelled and cancelled["status"] == "cancelled"
        assert cancelled["cancel_requested"] == 1
        assert cancelled["lease_owner"] is None
        assert jobs.cancellation_requested(conn, job["id"])
        assert not jobs.heartbeat_job(
            conn,
            job_id=job["id"],
            lease_owner="worker-a",
            now=BASE_TIME + timedelta(seconds=6),
        )
        jobs.request_cancellation(conn, job_id=job["id"], now=BASE_TIME + timedelta(seconds=7))
        assert [event["type"] for event in jobs.replay_events(conn, job_id=job["id"])] == [
            "terminal"
        ]
    finally:
        conn.close()


def test_events_are_ordered_replayable_and_terminal_transition_persists_usage() -> None:
    conn = _memory_conn()
    try:
        job, _ = _submit(conn)
        leased = jobs.lease_job(conn, job_id=job["id"], lease_owner="worker-a", now=BASE_TIME)
        assert leased
        jobs.append_event(
            conn,
            job_id=job["id"],
            event_type="candidate",
            payload={"reel_id": "provisional-1"},
            now=BASE_TIME + timedelta(seconds=1),
        )
        jobs.append_event(
            conn,
            job_id=job["id"],
            event_type="candidate",
            payload={"reel_id": "provisional-2"},
            now=BASE_TIME + timedelta(seconds=2),
        )
        jobs.append_event(
            conn,
            job_id=job["id"],
            event_type="final",
            payload={"reel_ids": ["final-1"]},
            now=BASE_TIME + timedelta(seconds=3),
        )
        jobs.record_provider_usage(
            conn,
            job_id=job["id"],
            provider="supadata",
            operation="search",
            billable_requests=1,
            now=BASE_TIME + timedelta(seconds=3),
        )
        terminal = jobs.transition_terminal(
            conn,
            job_id=job["id"],
            status="completed",
            result_generation_id="generation-1",
            lease_owner="worker-a",
            model_used="gemini-primary",
            quality_degraded=False,
            usage={"searches": 1, "transcripts": 1, "segmentations": 1},
            now=BASE_TIME + timedelta(seconds=4),
        )
        assert terminal and terminal["status"] == "completed"
        assert terminal["model_used"] == "gemini-primary"
        assert terminal["usage_json"] == '{"searches": 1, "transcripts": 1, "segmentations": 1}'

        all_events = jobs.replay_events(conn, job_id=job["id"])
        assert [event["seq"] for event in all_events] == [1, 2, 3, 4]
        assert [event["type"] for event in all_events] == [
            "candidate",
            "candidate",
            "final",
            "terminal",
        ]
        terminal_payload = all_events[-1]["payload"]
        assert terminal_payload["model_used"] == "gemini-primary"
        assert terminal_payload["quality_degraded"] is False
        assert terminal_payload["usage"] == {
            "searches": 1,
            "transcripts": 1,
            "segmentations": 1,
        }
        assert [event["seq"] for event in jobs.replay_events(conn, job_id=job["id"], after_seq=1)] == [
            2,
            3,
            4,
        ]
        assert all(
            event["job_id"] == job["id"] and event["timestamp"] for event in all_events
        )
        usage = conn.execute(
            "SELECT provider, operation, billable_requests FROM generation_provider_usage"
        ).fetchone()
        assert tuple(usage) == ("supadata", "search", 1)
    finally:
        conn.close()


class _RecordingCursor:
    def __init__(self, statements: list[str]) -> None:
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def execute(self, statement: str, _params=None) -> None:
        self.statements.append(" ".join(statement.split()))


class _RecordingPostgresConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def cursor(self):
        return _RecordingCursor(self.statements)


def test_postgres_schema_and_migration_sql_cover_durable_foundation() -> None:
    schema_sql = "\n".join(db._postgres_schema_statements())
    for table in (
        "generation_job_events",
        "supadata_search_cache",
        "transcript_artifacts",
        "generation_provider_usage",
        "concept_search_terms",
        "blocked_video_tombstones",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema_sql

    connection = _RecordingPostgresConnection()
    db._migrate_durable_generation_foundation_postgres(connection)
    migration_sql = "\n".join(connection.statements)
    assert "ADD COLUMN IF NOT EXISTS lease_owner TEXT" in migration_sql
    assert "ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0" in migration_sql
    assert "ADD COLUMN IF NOT EXISTS retry_not_before_at TEXT" in migration_sql
    assert "ALTER COLUMN max_attempts SET DEFAULT 3" in migration_sql
    assert "ADD COLUMN IF NOT EXISTS model_used TEXT" in migration_sql
    assert "Legacy refinement work was replaced by durable generation jobs" in migration_sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_generation_jobs_active_request" in migration_sql
    assert "WHERE status IN ('queued', 'running')" in migration_sql
