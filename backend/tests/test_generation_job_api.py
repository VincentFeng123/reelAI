from __future__ import annotations

import asyncio
import hashlib
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
from backend.app.clip_engine.errors import (
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderResponseValidationError,
    ProviderTransientError,
)
from backend.app.clip_engine.provider_runtime import ProviderUsageRecord
from backend.app.ingestion import pipeline as ingestion_pipeline
from backend.app.ingestion.persistence import ensure_clip_concept
from backend.app.ingestion.models import (
    IngestMetadata,
    IngestRequest,
    IngestSegment,
    IngestTranscriptCue,
    YouTubeSourceRef,
)
from backend.app.models import (
    AssessmentAnswerRequest,
    FeedbackRequest,
    MaterialLevelUpdateRequest,
    ReelOut,
    ReelsGenerateRequest,
)
from backend.app.services import generation_jobs, lesson_ordering
from backend.app.services.reels import ReelService
from backend.intent_obligations import (
    INTENT_OBLIGATION_CONTRACT_VERSION,
    intent_obligation,
)


EMPTY_ADAPTATION_FINGERPRINT = hashlib.sha256(b"{}").hexdigest()


class _PostgresTransactionFailure(RuntimeError):
    def __init__(self, sqlstate: str) -> None:
        self.sqlstate = sqlstate
        super().__init__(f"postgres transaction failure ({sqlstate})")


def test_generate_request_supports_full_material_inventory() -> None:
    assert ReelsGenerateRequest(material_id="m1").num_reels == 20
    assert ReelsGenerateRequest(material_id="m1", num_reels=300).num_reels == 300
    with pytest.raises(ValueError):
        ReelsGenerateRequest(material_id="m1", num_reels=301)


def test_fresh_inventory_and_selector_cache_share_current_contract() -> None:
    assert {
        main.SELECTION_CONTRACT_VERSION,
        ReelService.RANKED_FEED_CACHE_CONTRACT_VERSION,
    } == {"quality_silence_v39"}
    assert generation_jobs.REQUEST_SCHEMA_VERSION == "adaptive_clip_concepts_v4"
    assert segment_cache.SELECTION_CONTRACT_VERSION == "quality_silence_v39"
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
    assert "quality_silence_v28" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v29" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v30" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v31" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v32" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v37" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v38" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS
    assert "quality_silence_v39" in ReelService.DIFFICULTY_FALLBACK_CONTRACTS


def test_current_request_filter_rejects_stale_or_unversioned_inventory() -> None:
    current = {"reel_id": "current", "selection_contract_version": "quality_silence_v39"}
    internal_current = {
        "reel_id": "internal-current",
        "_selection_contract_version": "quality_silence_v39",
    }

    assert main._current_selection_contract_reels([
        {"reel_id": "stale", "selection_contract_version": "quality_silence_v36"},
        {"reel_id": "previous", "selection_contract_version": "quality_silence_v37"},
        {"reel_id": "old-adaptive", "selection_contract_version": "quality_silence_v38"},
        {"reel_id": "missing"},
        current,
        internal_current,
    ]) == [current, internal_current]


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


def test_intent_obligations_reach_organizer_only_from_trusted_gemini_metadata() -> None:
    obligation = intent_obligation(
        kind="scope",
        source_phrase="time and space complexity",
        requirement="Compare time and space complexity.",
        evidence_quote="Merge sort uses additional linear space.",
    )
    assert obligation is not None
    trusted = {
        "selection_contract_version": "quality_silence_v39",
        "selection_authority": "gemini",
        "intent_obligation_contract_version": (
            INTENT_OBLIGATION_CONTRACT_VERSION
        ),
        "intent_obligations": [obligation],
    }

    metadata = ReelService._selection_metadata(trusted)
    assert metadata["_selection_intent_obligations"] == [obligation]
    organizer_reel = main._public_generation_reel(
        {"reel_id": "trusted", **metadata},
        preserve_lesson_order_metadata=True,
    )
    assert organizer_reel["_selection_intent_obligations"] == [obligation]
    assert "_selection_intent_obligations" not in main._public_generation_reel(
        {"reel_id": "trusted", **metadata}
    )

    for untrusted in (
        {**trusted, "selection_authority": "local"},
        {**trusted, "intent_obligation_contract_version": "intent_obligation_v0"},
        {
            **trusted,
            "intent_obligations": [{**obligation, "key": "io:forged"}],
        },
        {
            **trusted,
            "intent_obligations": [
                {key: value for key, value in obligation.items() if key != "evidence_quote"}
            ],
        },
    ):
        assert "_selection_intent_obligations" not in (
            ReelService._selection_metadata(untrusted)
        )


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


def test_generation_db_transaction_retries_transient_body_on_fresh_checkout(
    monkeypatch,
) -> None:
    checkouts = [object(), object()]
    yielded: list[object] = []

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        current = checkouts[len(yielded)]
        yielded.append(current)
        yield current

    calls: list[object] = []

    def work(conn):
        calls.append(conn)
        if len(calls) == 1:
            raise _PostgresTransactionFailure("40001")
        return "stored"

    monkeypatch.setattr(main, "get_conn", connection)

    assert main._run_generation_db_transaction("test", work) == "stored"
    assert calls == checkouts
    assert yielded == checkouts


def test_generation_db_transaction_retries_when_lease_check_is_transient(
    monkeypatch,
) -> None:
    calls = 0
    stop_checks = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()

    def work(_conn):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _PostgresTransactionFailure("40001")
        return "stored"

    def retry_should_stop() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        raise _PostgresTransactionFailure("08006")

    monkeypatch.setattr(main, "get_conn", connection)

    assert main._run_generation_db_transaction(
        "test",
        work,
        retry_should_stop=retry_should_stop,
    ) == "stored"
    assert calls == 2
    assert stop_checks == 1


def test_generation_db_transaction_propagates_permanent_lease_check_failure(
    monkeypatch,
) -> None:
    calls = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()

    def work(_conn):
        nonlocal calls
        calls += 1
        raise _PostgresTransactionFailure("40001")

    def retry_should_stop() -> bool:
        raise _PostgresTransactionFailure("23505")

    monkeypatch.setattr(main, "get_conn", connection)

    with pytest.raises(_PostgresTransactionFailure, match="23505"):
        main._run_generation_db_transaction(
            "test",
            work,
            retry_should_stop=retry_should_stop,
        )
    assert calls == 1


def test_generation_db_transaction_retries_definite_commit_abort(
    monkeypatch,
) -> None:
    exits = 0
    calls = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal exits
        assert transactional is True
        yield object()
        exits += 1
        if exits == 1:
            raise _PostgresTransactionFailure("40P01")

    def work(_conn):
        nonlocal calls
        calls += 1
        return "released"

    monkeypatch.setattr(main, "get_conn", connection)

    assert main._run_generation_db_transaction("test", work) == "released"
    assert calls == 2
    assert exits == 2


def test_generation_db_transaction_does_not_retry_ambiguous_commit(
    monkeypatch,
) -> None:
    calls = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()
        raise _PostgresTransactionFailure("08006")

    def work(_conn):
        nonlocal calls
        calls += 1
        return "possibly-committed"

    monkeypatch.setattr(main, "get_conn", connection)

    with pytest.raises(_PostgresTransactionFailure, match="08006"):
        main._run_generation_db_transaction("test", work)
    assert calls == 1


def test_replay_safe_api_transaction_converges_after_lost_commit_ack(
    monkeypatch,
) -> None:
    calls = 0
    exits = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal exits
        assert transactional is True
        yield object()
        exits += 1
        if exits == 1:
            raise _PostgresTransactionFailure("08006")

    def work(_conn):
        nonlocal calls
        calls += 1
        return "converged"

    monkeypatch.setattr(main, "get_conn", connection)

    assert main._run_generation_db_transaction(
        "api-test",
        work,
        replay_after_unknown_commit=True,
    ) == "converged"
    assert calls == 2
    assert exits == 2


def test_replay_safe_api_transaction_does_not_retry_permanent_failure(
    monkeypatch,
) -> None:
    calls = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()

    def work(_conn):
        nonlocal calls
        calls += 1
        raise _PostgresTransactionFailure("23505")

    monkeypatch.setattr(main, "get_conn", connection)

    with pytest.raises(_PostgresTransactionFailure, match="23505"):
        main._run_generation_db_transaction(
            "api-test",
            work,
            replay_after_unknown_commit=True,
        )
    assert calls == 1


def test_generation_db_transaction_does_not_retry_permanent_failure(
    monkeypatch,
) -> None:
    calls = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()

    def work(_conn):
        nonlocal calls
        calls += 1
        raise _PostgresTransactionFailure("23505")

    monkeypatch.setattr(main, "get_conn", connection)

    with pytest.raises(_PostgresTransactionFailure, match="23505"):
        main._run_generation_db_transaction("test", work)
    assert calls == 1


def test_generation_db_transaction_stops_before_retry_when_lease_is_stale(
    monkeypatch,
) -> None:
    calls = 0
    stop_checks = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()

    def work(_conn):
        nonlocal calls
        calls += 1
        raise _PostgresTransactionFailure("40001")

    def retry_should_stop() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return True

    monkeypatch.setattr(main, "get_conn", connection)

    with pytest.raises(_PostgresTransactionFailure, match="40001"):
        main._run_generation_db_transaction(
            "test",
            work,
            retry_should_stop=retry_should_stop,
        )
    assert calls == 1
    assert stop_checks == 1


def test_generation_worker_retries_transient_job_lease_transaction(
    monkeypatch,
) -> None:
    checkouts = [object(), object()]
    lease_calls: list[object] = []
    run_calls: list[dict] = []
    stop = threading.Event()

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield checkouts[len(lease_calls)]

    def transient_lease(conn, **_kwargs):
        lease_calls.append(conn)
        if len(lease_calls) == 1:
            raise _PostgresTransactionFailure("40001")
        return {"id": "leased-after-retry"}

    def run_job(job, worker_stop):
        run_calls.append(job)
        assert worker_stop is stop
        stop.set()

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "lease_next_job", transient_lease)
    monkeypatch.setattr(main, "_run_leased_generation_job", run_job)

    main._generation_worker_loop("lease-retry-worker", stop)

    assert lease_calls == checkouts
    assert run_calls == [{"id": "leased-after-retry"}]


def test_generation_provider_usage_retries_once_with_stable_identity(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-usage-db-retry",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast"},
    )
    original_record = main.record_provider_usage
    usage_ids: list[str] = []

    def transient_then_store(worker_conn, **kwargs):
        usage_ids.append(str(kwargs["usage_id"]))
        if len(usage_ids) == 1:
            raise _PostgresTransactionFailure("40001")
        return original_record(worker_conn, **kwargs)

    monkeypatch.setattr(main, "record_provider_usage", transient_then_store)
    try:
        main._persist_generation_provider_usage(
            str(job["id"]),
            ProviderUsageRecord(
                provider="gemini",
                operation="segmentation",
                attempt=1,
                timestamp="2026-07-20T00:00:00+00:00",
                billable_requests=1,
                model_used="gemini-test",
            ),
        )

        assert len(usage_ids) == 2
        assert usage_ids[0] == usage_ids[1]
        assert conn.execute(
            "SELECT COUNT(*) FROM generation_provider_usage WHERE job_id = ?",
            (job["id"],),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_generation_gemini_ledger_exposure_unions_and_deduplicates_snapshots() -> None:
    conn = _conn()
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-usage-union",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast"},
    )
    base_metadata = {
        "provider_call": True,
        "billing_usage_known": True,
        "cached_tokens": 0,
        "reserved_cost_usd": 0.25,
        "admitted_cost_usd": 0.25,
    }
    prior_record = {
        "provider": "gemini",
        "operation": "segmentation",
        "attempt": 1,
        "timestamp": "2026-07-20T00:00:00+00:00",
        "status_code": 200,
        "billable_requests": 1,
        "input_tokens": 100_000,
        "output_tokens": 0,
        "total_tokens": 100_000,
        "model_used": "gemini-3.1-pro-preview",
        "quality_degraded": False,
        "error_code": "",
        "metadata": dict(base_metadata),
    }
    prior_only_record = {
        **prior_record,
        "timestamp": "2026-07-19T23:59:59+00:00",
        "metadata": dict(base_metadata),
    }
    for timestamp in (
        "2026-07-20T00:00:00+00:00",
        "2026-07-20T00:00:01+00:00",
    ):
        generation_jobs.record_provider_usage(
            conn,
            job_id=str(job["id"]),
            provider="gemini",
            operation="segmentation",
            model="gemini-3.1-pro-preview",
            billable_requests=1,
            input_tokens=100_000,
            output_tokens=0,
            total_tokens=100_000,
            metadata={
                **base_metadata,
                "attempt": 1,
                "status_code": 200,
                "quality_degraded": False,
                "error_code": "",
                "timestamp": timestamp,
            },
            now=timestamp,
        )
    try:
        exposure = main._generation_gemini_ledger_exposure(
            conn,
            str(job["id"]),
            prior_records=[prior_only_record, prior_record],
        )

        assert exposure["committed_cost_usd"] == pytest.approx(0.6)
        assert exposure["cost_exposure_usd"] == pytest.approx(0.6)
        assert exposure["lifetime_reserved_worst_case_cost_usd"] == pytest.approx(
            0.75
        )
    finally:
        conn.close()


def test_ambiguous_final_release_commit_is_not_replayed_or_duplicated(
    monkeypatch,
) -> None:
    conn = _conn()
    now = datetime.now(timezone.utc)
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="ambiguous-release",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="ambiguous-release",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast"},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="ambiguous-release-worker",
        now=now,
    )
    assert leased
    commit_reports = 0
    release_calls = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal commit_reports
        assert transactional is True
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
            commit_reports += 1
            if commit_reports == 1:
                raise _PostgresTransactionFailure("08006")

    def release(worker_conn):
        nonlocal release_calls
        release_calls += 1
        main._activate_generation(
            worker_conn,
            material_id="m1",
            request_key="ambiguous-release",
            generation_id=generation_id,
            retrieval_profile="unified",
        )
        generation_jobs.append_event(
            worker_conn,
            job_id=str(job["id"]),
            event_type="final",
            payload={
                "reels": [],
                "generation_id": generation_id,
                "authoritative": True,
            },
            lease_owner="ambiguous-release-worker",
        )
        generation_jobs.transition_terminal(
            worker_conn,
            job_id=str(job["id"]),
            status="partial",
            result_generation_id=generation_id,
            lease_owner="ambiguous-release-worker",
            usage={},
        )

    monkeypatch.setattr(main, "get_conn", connection)
    try:
        with pytest.raises(_PostgresTransactionFailure, match="08006"):
            main._run_generation_db_transaction("final_release", release)

        recovered = generation_jobs.transition_terminal(
            conn,
            job_id=str(job["id"]),
            status="failed",
            result_generation_id=generation_id,
            lease_owner="ambiguous-release-worker",
            usage={},
        )
        assert recovered is not None
        assert recovered["status"] == "partial"
        assert release_calls == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_heads WHERE active_generation_id = ?",
            (generation_id,),
        ).fetchone()[0] == 1
        assert [
            event["type"]
            for event in generation_jobs.replay_events(conn, job_id=str(job["id"]))
        ] == ["final", "terminal"]
    finally:
        conn.close()


def test_semantic_family_metadata_does_not_expand_exact_organizer_signals() -> None:
    conn = _conn()
    learner_id = "owner:semantic-family"
    try:
        db._migrate_reel_feedback_uniqueness_sqlite(conn)
        conn.execute("UPDATE concepts SET title = ? WHERE id = 'c1'", ("Newton's first law",))
        source_context = json.dumps({
            "selection_contract_version": "quality_silence_v39",
            "selection_authority": "gemini",
            "concept_family_contract_version": "concept_family_v3",
            "concept_family": "Newton's first law",
            "concept_aliases": [],
        })
        conn.execute(
            "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
            "VALUES ('family-video', 'Action reaction', 'channel', 300, "
            "'2026-07-10T00:00:01+00:00')"
        )
        conn.execute(
            "INSERT INTO reels "
            "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
            "transcript_snippet, takeaways_json, base_score, difficulty, created_at, "
            "search_context_json) "
            "VALUES ('family-reel', 'm1', 'c1', 'family-video', '', 0, 20, "
            "'An object keeps its state of motion unless a net force acts.', '[]', "
            "1.0, 0.5, '2026-07-10T00:00:02+00:00', ?)",
            (source_context,),
        )
        main.reel_service.record_feedback(
            conn,
            "family-reel",
            helpful=True,
            confusing=False,
            rating=None,
            saved=False,
            learner_id=learner_id,
        )
        before = main._learner_adaptation_fingerprint(
            conn, material_id="m1", learner_id=learner_id
        )

        related_id, _, _ = ensure_clip_concept(
            conn,
            material_id="m1",
            title="inertia and motion",
        )
        target_context = json.dumps({
            "selection_contract_version": "quality_silence_v39",
            "selection_authority": "gemini",
            "concept_family_contract_version": "concept_family_v3",
            "concept_family": "Newton's first law",
            "concept_aliases": [],
            "clip_concept_raw": "inertia resists changes in motion",
        })
        conn.execute(
            "INSERT INTO reels "
            "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
            "transcript_snippet, takeaways_json, base_score, difficulty, created_at, "
            "search_context_json) VALUES (?, 'm1', ?, 'family-video', '', 30, 50, "
            "'Inertia resists changes to an object state of motion.', '[]', 1.0, "
            "0.4, '2026-07-10T00:00:03+00:00', ?)",
            ("new-family-reel", related_id, target_context),
        )

        assert main._learner_adaptation_fingerprint(
            conn, material_id="m1", learner_id=learner_id
        ) == before
        signals = main._learner_concept_signals(
            conn, material_id="m1", learner_id=learner_id
        )
        assert signals["c1"]["helpful"] == 1.0
        assert related_id not in signals
        ranked = main.reel_service.ranked_feed(
            conn,
            material_id="m1",
            learner_id=learner_id,
        )
        target_reel = next(
            reel for reel in ranked if reel["reel_id"] == "new-family-reel"
        )
        organizer_reel = main._public_generation_reel(
            target_reel,
            preserve_lesson_order_metadata=True,
        )
        assert (
            organizer_reel["_selection_concept"]
            == "inertia resists changes in motion"
        )
        assert "_selection_concept" not in main._public_generation_reel(target_reel)
        prompt = lesson_ordering._user_prompt(
            [{
                **organizer_reel,
                "concept_id": related_id,
            }],
            topic="Newton's laws",
            learner_level="beginner",
            concept_signals=signals,
        )
        prompt_payload = json.loads(
            prompt.split("CLIPS_JSON:\n", 1)[1].split("\n\nFinal request:", 1)[0]
        )
        organizer_clip = prompt_payload["clips"][0]
        assert organizer_clip["concept_title"] == "inertia resists changes in motion"
        assert organizer_clip["concept_family"] == "Newton's first law"
        assert organizer_clip["concept_aliases"] == []
        assert organizer_clip["learner_signal"] == {
            "helpful": 0.0,
            "confusing": 0.0,
            "adjustment": 0.0,
        }

        main.reel_service.record_feedback(
            conn,
            "family-reel",
            helpful=False,
            confusing=True,
            rating=None,
            saved=False,
            learner_id=learner_id,
        )
        remediation_concept_ids: list[str] = []
        confusing_signals = main._learner_concept_signals(
            conn,
            material_id="m1",
            learner_id=learner_id,
            remediation_concept_ids_out=remediation_concept_ids,
        )
        assert confusing_signals["c1"]["confusing"] == 1.0
        assert related_id not in confusing_signals
        assert remediation_concept_ids == ["c1"]
    finally:
        conn.close()


def test_organizer_adaptation_ignores_explicit_stale_family_provenance() -> None:
    conn = _conn()
    learner_id = "owner:family-rollout"
    try:
        db._migrate_reel_feedback_uniqueness_sqlite(conn)
        for video_id in ("stale-video", "current-video", "legacy-video"):
            conn.execute(
                "INSERT INTO videos (id, title, channel_title, duration_sec, "
                "created_at) VALUES (?, ?, 'channel', 300, "
                "'2026-07-10T00:00:01+00:00')",
                (video_id, video_id),
            )
        contexts = {
            "stale-reel": json.dumps({
                "selection_contract_version": "quality_silence_v38",
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v2",
                "concept_family": "cellular respiration",
                "concept_aliases": [],
            }),
            "current-reel": json.dumps({
                "selection_contract_version": "quality_silence_v39",
                "selection_authority": "gemini",
                "concept_family_contract_version": "concept_family_v3",
                "concept_family": "mitochondrial ATP production",
                "concept_aliases": [],
                "clip_concept_raw": "mitochondrial ATP production",
            }),
            "legacy-reel": "{}",
        }
        for index, (reel_id, video_id) in enumerate((
            ("stale-reel", "stale-video"),
            ("current-reel", "current-video"),
            ("legacy-reel", "legacy-video"),
        )):
            conn.execute(
                "INSERT INTO reels (id, material_id, concept_id, video_id, "
                "video_url, t_start, t_end, transcript_snippet, takeaways_json, "
                "base_score, difficulty, created_at, search_context_json) "
                "VALUES (?, 'm1', 'c1', ?, '', ?, ?, 'ATP explanation', '[]', "
                "1.0, 0.5, '2026-07-10T00:00:02+00:00', ?)",
                (reel_id, video_id, index * 30, index * 30 + 20, contexts[reel_id]),
            )
        for reel_id, helpful, confusing in (
            ("stale-reel", False, True),
            ("current-reel", True, False),
            ("legacy-reel", False, True),
        ):
            main.reel_service.record_feedback(
                conn,
                reel_id,
                helpful=helpful,
                confusing=confusing,
                rating=None,
                saved=False,
                learner_id=learner_id,
            )

        signals = main._learner_concept_signals(
            conn,
            material_id="m1",
            learner_id=learner_id,
        )
        assert signals["c1"]["helpful"] == 1.0
        assert signals["c1"]["confusing"] == 1.0
        assert signals["c1"]["adjustment"] == pytest.approx(-0.02)
        prompt = lesson_ordering._user_prompt(
            [{
                "reel_id": "current-reel",
                "concept_id": "c1",
                "_selection_concept": "mitochondrial ATP production",
            }],
            topic="cell biology",
            learner_level="beginner",
            concept_signals=signals,
        )
        prompt_payload = json.loads(
            prompt.split("CLIPS_JSON:\n", 1)[1].split("\n\nFinal request:", 1)[0]
        )
        assert prompt_payload["clips"][0]["learner_signal"] == signals["c1"]

        assert main._lesson_prior_concept_coverage(
            conn,
            material_id="m1",
            reel_ids=["stale-reel"],
        ) == []
        prior = main._lesson_prior_concept_coverage(
            conn,
            material_id="m1",
            reel_ids=["stale-reel", "current-reel", "legacy-reel"],
        )
        assert sum(int(row["delivered_count"]) for row in prior) == 2
    finally:
        conn.close()


def test_lesson_prior_coverage_unions_trusted_intent_obligation_keys() -> None:
    conn = _conn()
    obligations = [
        intent_obligation(
            kind="subject",
            source_phrase=source_phrase,
            requirement=requirement,
            evidence_quote=evidence,
        )
        for source_phrase, requirement, evidence in (
            ("bubble sort", "Teach bubble sort.", "Bubble sort swaps adjacent values."),
            ("merge sort", "Teach merge sort.", "Merge sort combines sorted halves."),
            ("quick sort", "Teach quick sort.", "Quick sort partitions around a pivot."),
            ("heap sort", "Teach heap sort.", "Heap sort repeatedly extracts a maximum."),
        )
    ]
    assert all(obligations)
    trusted_batches = [
        [obligations[0], obligations[1]],
        [obligations[1], obligations[2]],
    ]
    try:
        for index, batch in enumerate(trusted_batches):
            reel_id = f"trusted-obligation-{index}"
            video_id = f"trusted-obligation-video-{index}"
            conn.execute(
                "INSERT INTO videos (id, title, channel_title, duration_sec, "
                "created_at) VALUES (?, ?, 'Test', 120, "
                "'2026-07-20T00:00:00+00:00')",
                (video_id, video_id),
            )
            conn.execute(
                "INSERT INTO reels (id, material_id, concept_id, video_id, "
                "video_url, t_start, t_end, transcript_snippet, takeaways_json, "
                "base_score, difficulty, search_context_json, created_at) "
                "VALUES (?, 'm1', 'c1', ?, '', 0, 30, 'Sorting explanation', "
                "'[]', 1, 0.1, ?, '2026-07-20T00:00:00+00:00')",
                (
                    reel_id,
                    video_id,
                    json.dumps({
                        "selection_contract_version": "quality_silence_v39",
                        "selection_authority": "gemini",
                        "intent_obligation_contract_version": (
                            INTENT_OBLIGATION_CONTRACT_VERSION
                        ),
                        "intent_obligations": batch,
                    }),
                ),
            )

        untrusted_video_id = "untrusted-obligation-video"
        conn.execute(
            "INSERT INTO videos (id, title, channel_title, duration_sec, "
            "created_at) VALUES (?, ?, 'Test', 120, "
            "'2026-07-20T00:00:00+00:00')",
            (untrusted_video_id, untrusted_video_id),
        )
        conn.execute(
            "INSERT INTO reels (id, material_id, concept_id, video_id, video_url, "
            "t_start, t_end, transcript_snippet, takeaways_json, base_score, "
            "difficulty, search_context_json, created_at) VALUES "
            "('untrusted-obligation', 'm1', 'c1', ?, '', 0, 30, "
            "'Sorting explanation', '[]', 1, 0.1, ?, "
            "'2026-07-20T00:00:00+00:00')",
            (
                untrusted_video_id,
                json.dumps({
                    "selection_contract_version": "quality_silence_v39",
                    "selection_authority": "local",
                    "intent_obligation_contract_version": (
                        INTENT_OBLIGATION_CONTRACT_VERSION
                    ),
                    "intent_obligations": [obligations[3]],
                }),
            ),
        )

        prior = main._lesson_prior_concept_coverage(
            conn,
            material_id="m1",
            reel_ids=[
                "trusted-obligation-0",
                "trusted-obligation-1",
                "untrusted-obligation",
            ],
        )

        assert len(prior) == 1
        assert prior[0]["delivered_count"] == 3
        assert prior[0]["intent_obligation_keys"] == sorted({
            obligations[0]["key"],
            obligations[1]["key"],
            obligations[2]["key"],
        })
        assert obligations[3]["key"] not in prior[0]["intent_obligation_keys"]
    finally:
        conn.close()


def test_lesson_prior_coverage_does_not_truncate_late_obligation_witness() -> None:
    conn = _conn()
    obligation = intent_obligation(
        kind="scope",
        source_phrase="late requested facet",
        requirement="Teach the late requested facet",
        evidence_quote="This clip teaches the late requested facet clearly",
    )
    assert obligation is not None
    try:
        conn.execute(
            "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
            "VALUES ('history-video', 'History', 'Test', 120, "
            "'2026-07-20T00:00:00+00:00')"
        )
        ordinary_context = json.dumps({
            "selection_contract_version": "quality_silence_v39",
            "selection_authority": "gemini",
        })
        reel_ids = [f"aa-history-{index:03d}" for index in range(204)]
        reel_ids.append("zz-late-obligation-witness")
        for reel_id in reel_ids[:-1]:
            conn.execute(
                "INSERT INTO reels (id, material_id, concept_id, video_id, "
                "video_url, t_start, t_end, transcript_snippet, takeaways_json, "
                "base_score, difficulty, search_context_json, created_at) VALUES "
                "(?, 'm1', 'c1', 'history-video', '', 0, 20, 'History', '[]', "
                "1, 0.1, ?, '2026-07-20T00:00:00+00:00')",
                (reel_id, ordinary_context),
            )
        conn.execute(
            "INSERT INTO reels (id, material_id, concept_id, video_id, video_url, "
            "t_start, t_end, transcript_snippet, takeaways_json, base_score, "
            "difficulty, search_context_json, created_at) VALUES "
            "('zz-late-obligation-witness', 'm1', 'c1', 'history-video', '', "
            "20, 40, 'Late facet', '[]', 1, 0.1, ?, "
            "'2026-07-20T00:00:00+00:00')",
            (json.dumps({
                "selection_contract_version": "quality_silence_v39",
                "selection_authority": "gemini",
                "intent_obligation_contract_version": (
                    INTENT_OBLIGATION_CONTRACT_VERSION
                ),
                "intent_obligations": [obligation],
            }),),
        )

        prior = main._lesson_prior_concept_coverage(
            conn,
            material_id="m1",
            reel_ids=reel_ids,
        )

        assert len(prior) == 1
        assert prior[0]["delivered_count"] == 205
        assert prior[0]["intent_obligation_keys"] == [obligation["key"]]
    finally:
        conn.close()


def test_lesson_prior_coverage_keeps_narrow_concepts_in_one_family_separate() -> None:
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO concepts (id, material_id, title, keywords_json, summary, "
            "created_at) VALUES ('c2', 'm1', 'Mass and required force', '[]', '', "
            "'2026-07-20T00:00:00+00:00')"
        )
        context = json.dumps({
            "selection_contract_version": "quality_silence_v39",
            "selection_authority": "gemini",
            "concept_family_contract_version": "concept_family_v3",
            "concept_family": "Newton's second law of motion",
            "concept_aliases": [],
        })
        for index, concept_id in enumerate(("c1", "c2")):
            video_id = f"narrow-family-video-{index}"
            reel_id = f"narrow-family-reel-{index}"
            conn.execute(
                "INSERT INTO videos (id, title, channel_title, duration_sec, "
                "created_at) VALUES (?, ?, 'Test', 120, "
                "'2026-07-20T00:00:00+00:00')",
                (video_id, video_id),
            )
            conn.execute(
                "INSERT INTO reels (id, material_id, concept_id, video_id, "
                "video_url, t_start, t_end, transcript_snippet, takeaways_json, "
                "base_score, difficulty, search_context_json, created_at) VALUES "
                "(?, 'm1', ?, ?, '', ?, ?, 'Narrow teaching', '[]', 1, 0.2, ?, "
                "'2026-07-20T00:00:00+00:00')",
                (
                    reel_id,
                    concept_id,
                    video_id,
                    index * 30,
                    index * 30 + 20,
                    context,
                ),
            )

        prior = main._lesson_prior_concept_coverage(
            conn,
            material_id="m1",
            reel_ids=["narrow-family-reel-0", "narrow-family-reel-1"],
        )

        assert len(prior) == 2
        assert {item["concept_id"] for item in prior} == {"c1", "c2"}
        assert all(item["delivered_count"] == 1 for item in prior)
        assert {item["concept_family"] for item in prior} == {
            "Newton's second law of motion"
        }
    finally:
        conn.close()


def test_persisted_lesson_order_reapplies_a_valid_selected_subset() -> None:
    conn = _conn()
    try:
        generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key="lesson-order-exact-set",
            generation_mode="fast",
            retrieval_profile="unified",
        )
        reels = [{"reel_id": "a"}, {"reel_id": "b"}]
        main._persist_generation_lesson_order(
            conn,
            generation_id=generation_id,
            metadata={"version": 2, "ordered_reel_ids": ["b", "a"]},
        )
        assert main._apply_generation_lesson_order(
            conn, generation_id=generation_id, reels=reels
        ) == [reels[1], reels[0]]

        main._persist_generation_lesson_order(
            conn,
            generation_id=generation_id,
            metadata={"version": 2, "ordered_reel_ids": ["a"]},
        )
        assert main._apply_generation_lesson_order(
            conn, generation_id=generation_id, reels=reels
        ) == [reels[0]]

        main._persist_generation_lesson_order(
            conn,
            generation_id=generation_id,
            metadata={"version": 2, "ordered_reel_ids": ["unknown"]},
        )
        assert main._apply_generation_lesson_order(
            conn, generation_id=generation_id, reels=reels
        ) == reels

        main._persist_generation_lesson_order(
            conn,
            generation_id=generation_id,
            metadata={"version": 2, "ordered_reel_ids": []},
        )
        assert main._apply_generation_lesson_order(
            conn, generation_id=generation_id, reels=reels
        ) == reels
    finally:
        conn.close()


def test_persisted_lesson_order_projects_onto_remaining_unseen_reels() -> None:
    conn = _conn()
    try:
        generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key="lesson-order-filtered-subset",
            generation_mode="fast",
            retrieval_profile="unified",
        )
        reels = [{"reel_id": "a"}, {"reel_id": "b"}, {"reel_id": "c"}]
        main._persist_generation_lesson_order(
            conn,
            generation_id=generation_id,
            metadata={"version": 2, "ordered_reel_ids": ["a", "b", "c"]},
        )

        assert main._apply_generation_lesson_order(
            conn,
            generation_id=generation_id,
            reels=[reels[2], reels[1]],
        ) == [reels[1], reels[2]]
        assert main._apply_generation_lesson_order(
            conn,
            generation_id=generation_id,
            reels=[],
        ) == []

        unknown = {"reel_id": "unknown"}
        assert main._apply_generation_lesson_order(
            conn,
            generation_id=generation_id,
            reels=[reels[1], unknown],
        ) == [reels[1], unknown]
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("checkpoint_ids", "preparation_complete"),
    [
        (["release-reel-1"], True),
        ([], True),
        (["release-reel-1"], False),
    ],
)
def test_generation_prepares_only_organizer_checkpoints_before_release(
    monkeypatch,
    checkpoint_ids,
    preparation_complete,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    suffix = (
        "empty"
        if not checkpoint_ids
        else "selected"
        if preparation_complete
        else "incomplete"
    )
    concept_signals = {
        "c1": {"helpful": 1.0, "confusing": 0.0, "adjustment": 0.04}
    }
    adaptation_fingerprint = hashlib.sha256(
        json.dumps(concept_signals, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"lesson-order-release-{suffix}",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 3,
            "knowledge_level": "beginner",
            "adaptation_fingerprint": adaptation_fingerprint,
        },
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner=f"lesson-order-worker-{suffix}",
        now=now,
    )
    assert leased
    generated: list[dict] = []

    def generate_stage(worker_conn, **kwargs) -> None:
        for index in range(3):
            generated.append(
                _insert_generation_reel(
                    worker_conn,
                    generation_id=str(kwargs["generation_id"]),
                    reel_id=f"release-reel-{index}",
                    video_id=f"release-video-{index}",
                    created_at=(now + timedelta(seconds=index)).isoformat(),
                )
            )

    def order_batch(reels, **kwargs):
        assert kwargs["topic"] == "Cell biology"
        assert kwargs["concept_signals"] == {
            "c1": {"helpful": 1.0, "confusing": 0.0, "adjustment": 0.04}
        }
        ordered = list(reversed(reels))
        return mock.Mock(
            reels=ordered,
            ordered_reel_ids=[reel["reel_id"] for reel in ordered],
            assessment_checkpoint_reel_ids=checkpoint_ids,
            terminal_summary_start_reel_id="release-reel-1",
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    original_prepare = main.assessment_service.prepare_reel_questions
    preparation_calls: list[dict] = []

    def prepare_before_release(worker_conn, **kwargs):
        preparation_calls.append(kwargs)
        assert kwargs["reel_ids"] == ["release-reel-1"]
        assert kwargs["use_model"] is False
        assert generation_jobs.replay_events(conn, job_id=job["id"]) == []
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_heads"
        ).fetchone()[0] == 0
        if not preparation_complete:
            return {"requested": 1, "prepared": 0, "fallback": 0}
        return original_prepare(worker_conn, **kwargs)

    original_append = main.append_generation_event

    def append_after_preparation(worker_conn, **kwargs):
        if checkpoint_ids and preparation_complete:
            assert conn.execute(
                "SELECT COUNT(*) FROM reel_assessment_questions"
            ).fetchone()[0] == 1
        return original_append(worker_conn, **kwargs)

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_learner_concept_signals",
        lambda *_args, **_kwargs: concept_signals,
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(generated),
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    monkeypatch.setattr(
        main.assessment_service,
        "prepare_reel_questions",
        prepare_before_release,
    )
    monkeypatch.setattr(main, "append_generation_event", append_after_preparation)

    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed = generation_jobs.get_job(conn, job["id"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert len(preparation_calls) == (1 if checkpoint_ids else 0)
        generation_row = main._fetch_generation_row(
            conn, str(completed["result_generation_id"])
        )
        metadata = json.loads(str(generation_row["lesson_order_json"]))
        assert metadata["ordered_reel_ids"] == [
            "release-reel-2",
            "release-reel-1",
            "release-reel-0",
        ]
        assert metadata["assessment_checkpoint_reel_ids"] == (
            checkpoint_ids if preparation_complete else None
        )
        assert metadata["terminal_summary_start_reel_id"] == "release-reel-1"
        assert metadata["degraded"] is (not preparation_complete)
        if not preparation_complete:
            assert metadata["fallback_reason"] == "recall_preparation_unavailable"
            assert completed["quality_degraded"] == 1
        final = next(
            event
            for event in generation_jobs.replay_events(conn, job_id=job["id"])
            if event["type"] == "final"
        )
        assert [reel["reel_id"] for reel in final["payload"]["reels"]] == [
            "release-reel-2",
            "release-reel-1",
            "release-reel-0",
        ]
    finally:
        conn.close()


def test_generation_organizer_can_select_any_subset_from_a_larger_candidate_window(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="lesson-order-candidate-source",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    candidate_rows = [
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"window-reel-{index}",
            video_id=f"window-video-{index}",
            created_at=(now + timedelta(seconds=index)).isoformat(),
        )
        for index in range(9)
    ]
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="lesson-order-prior-release",
        generation_id=source_generation_id,
        completed_at=(now - timedelta(seconds=1)).isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "fast", "num_reels": 3},
    )
    _append_authoritative_release(
        conn,
        job_id=str(prior_job["id"]),
        reel_ids=[f"window-reel-{index}" for index in range(3)],
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="lesson-order-candidate-window",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 3,
            "knowledge_level": "beginner",
            "continuation_token": prior_job["id"],
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="lesson-window-worker",
        now=now,
    )
    assert leased
    overlap_filter_calls: list[list[str]] = []
    original_overlap_filter = main._filter_continuation_release_temporal_overlaps

    def filter_before_final(worker_conn, **kwargs):
        overlap_filter_calls.append([
            str(reel.get("reel_id") or "") for reel in kwargs["reels"]
        ])
        return original_overlap_filter(worker_conn, **kwargs)

    def order_batch(reels, **kwargs):
        assert [reel["reel_id"] for reel in reels] == [
            f"window-reel-{index}" for index in range(3, 9)
        ]
        assert kwargs["release_limit"] == len(reels) == 6
        assert kwargs["prior_concept_coverage"] == [{
            "concept_id": "c1",
            "concept_family": "",
            "concept_title": "Mitochondria",
            "delivered_count": 3,
        }]
        selected = [reels[0], reels[1], reels[2], reels[4], reels[5]]
        return mock.Mock(
            reels=selected,
            ordered_reel_ids=[reel["reel_id"] for reel in selected],
            assessment_checkpoint_reel_ids=[],
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    monkeypatch.setattr(
        main.reel_service,
        "generate_reels",
        mock.Mock(side_effect=AssertionError("cached reservoir must not retrieve")),
    )
    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 3,
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(candidate_rows[3:]),
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    monkeypatch.setattr(
        main,
        "_filter_continuation_release_temporal_overlaps",
        filter_before_final,
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert overlap_filter_calls == [[
            f"window-reel-{index}" for index in range(3, 9)
        ]]

        final = next(
            event
            for event in generation_jobs.replay_events(conn, job_id=job["id"])
            if event["type"] == "final"
        )
        assert [reel["reel_id"] for reel in final["payload"]["reels"]] == [
            "window-reel-3",
            "window-reel-4",
            "window-reel-5",
            "window-reel-7",
            "window-reel-8",
        ]
        completed = generation_jobs.get_job(conn, job["id"])
        assert completed is not None
        expected_ids = [
            "window-reel-3",
            "window-reel-4",
            "window-reel-5",
            "window-reel-7",
            "window-reel-8",
        ]
        status_payload = main._generation_job_status_payload(conn, completed)
        assert [reel["reel_id"] for reel in status_payload["reels"]] == expected_ids
        replay = main._sanitize_generation_replay_events(
            conn,
            completed,
            generation_jobs.replay_events(conn, job_id=job["id"]),
        )
        sanitized_final = next(event for event in replay if event["type"] == "final")
        assert [
            reel["reel_id"] for reel in sanitized_final["payload"]["reels"]
        ] == expected_ids
    finally:
        conn.close()


def test_continuation_prefilters_prior_overlap_before_capacity_limited_ordering(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="prefilter-overlap-source",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    parent = _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="released-parent",
        video_id="shared-video",
        created_at=now.isoformat(),
    )
    overlap = _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="overlapping-child",
        video_id="temporary-overlap-video",
        created_at=(now + timedelta(seconds=1)).isoformat(),
    )
    conn.execute(
        "UPDATE reels SET video_id = 'shared-video', t_start = 5, t_end = 25 "
        "WHERE id = 'overlapping-child'"
    )
    overlap.update(video_id="shared-video", t_start=5.0, t_end=25.0)
    alternative = _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="novel-mandatory-alternative",
        video_id="novel-video",
        created_at=(now + timedelta(seconds=2)).isoformat(),
    )
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="prefilter-overlap-prior",
        generation_id=source_generation_id,
        completed_at=(now - timedelta(seconds=1)).isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "fast", "num_reels": 1},
    )
    _append_authoritative_release(
        conn,
        job_id=str(prior_job["id"]),
        reel_ids=[parent["reel_id"]],
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="prefilter-overlap-child",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 1,
            "knowledge_level": "beginner",
            "continuation_token": prior_job["id"],
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="prefilter-overlap-worker",
        now=now,
    )
    assert leased

    def order_one(reels, **kwargs):
        assert [reel["reel_id"] for reel in reels] == [
            "novel-mandatory-alternative"
        ]
        assert kwargs["release_limit"] == 1
        selected = list(reels[:1])
        return mock.Mock(
            reels=selected,
            ordered_reel_ids=[reel["reel_id"] for reel in selected],
            assessment_checkpoint_reel_ids=[],
            terminal_summary_start_reel_id=None,
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    monkeypatch.setattr(
        main.reel_service,
        "generate_reels",
        mock.Mock(side_effect=AssertionError("cached reservoir must not retrieve")),
    )
    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [overlap, alternative],
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_one)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        final = next(
            event
            for event in generation_jobs.replay_events(conn, job_id=job["id"])
            if event["type"] == "final"
        )
        assert [reel["reel_id"] for reel in final["payload"]["reels"]] == [
            "novel-mandatory-alternative"
        ]
    finally:
        conn.close()


def test_continuation_with_only_prior_overlaps_does_not_activate_empty_order(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="all-overlap-source",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    parent = _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="all-overlap-parent",
        video_id="all-overlap-video",
        created_at=now.isoformat(),
    )
    overlap = _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="all-overlap-child",
        video_id="temporary-all-overlap-video",
        created_at=(now + timedelta(seconds=1)).isoformat(),
    )
    conn.execute(
        "UPDATE reels SET video_id = 'all-overlap-video', t_start = 5, t_end = 25 "
        "WHERE id = 'all-overlap-child'"
    )
    overlap.update(video_id="all-overlap-video", t_start=5.0, t_end=25.0)
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="all-overlap-prior",
        generation_id=source_generation_id,
        completed_at=(now - timedelta(seconds=1)).isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "fast", "num_reels": 1},
    )
    _append_authoritative_release(
        conn,
        job_id=str(prior_job["id"]),
        reel_ids=[parent["reel_id"]],
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="all-overlap-continuation",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 1,
            "knowledge_level": "beginner",
            "continuation_token": prior_job["id"],
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="all-overlap-worker",
        now=now,
    )
    assert leased
    activate = mock.Mock()
    monkeypatch.setattr(
        main.reel_service,
        "generate_reels",
        mock.Mock(side_effect=AssertionError("cached reservoir must not retrieve")),
    )
    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [overlap],
    )
    monkeypatch.setattr(
        main,
        "order_lesson_batch",
        mock.Mock(side_effect=AssertionError("empty pool must not be ordered")),
    )
    monkeypatch.setattr(main, "_activate_generation", activate)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed = generation_jobs.get_job(conn, job["id"])
        assert completed is not None
        assert completed["status"] == "exhausted"
        final = next(
            event
            for event in generation_jobs.replay_events(conn, job_id=job["id"])
            if event["type"] == "final"
        )
        assert final["payload"]["reels"] == []
        assert final["payload"]["generation_id"] is None
        child_row = conn.execute(
            "SELECT status, activated_at, lesson_order_json FROM reel_generations "
            "WHERE source_generation_id = ? ORDER BY created_at DESC LIMIT 1",
            (source_generation_id,),
        ).fetchone()
        assert child_row is not None
        assert tuple(child_row) == ("failed", None, None)
        activate.assert_not_called()
    finally:
        conn.close()


def test_fresh_generation_gives_organizer_every_candidate_from_bounded_work(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="fresh-organizer-parent-window",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    clip_lessons = [
        (
            "Cell biology explains mitochondrial energy mechanism "
            f"marker {index} with a distinct complete teaching example."
        )
        for index in range(128)
    ]

    def configure_window_reel(
        worker_conn,
        *,
        index: int,
        source_video_id: str,
        chain_id: str,
        chain_position: int,
        prerequisite_ids: list[str],
    ) -> None:
        start = float(chain_position * 31)
        end = start + 30.0
        context = json.loads(worker_conn.execute(
            "SELECT search_context_json FROM reels WHERE id = ?",
            (f"fresh-window-reel-{index}",),
        ).fetchone()[0])
        context["concept_family"] = "mitochondrial oxidative phosphorylation"
        context["concept_aliases"] = []
        context["concept_family_contract_version"] = "concept_family_v3"
        context["selection_authority"] = "gemini"
        context["source_rank"] = index % 3
        context["selection_candidate_id"] = f"candidate-{index}"
        context["chain_id"] = chain_id
        context["chain_position"] = chain_position
        context["prerequisite_ids"] = prerequisite_ids
        context["topic_evidence_quote"] = clip_lessons[index]
        context["boundary_diagnostics"]["final_range"] = [start, end]
        context["boundary_diagnostics"]["acoustic"] = {
            "threshold_dbfs": -38.0,
            "start_quiet": [max(0.0, start - 0.1), start + 0.1],
            "end_quiet": [end - 0.1, end + 0.1],
        }
        worker_conn.execute(
            "UPDATE reels SET video_id = ?, t_start = ?, t_end = ?, "
            "transcript_snippet = ?, ai_summary = ?, takeaways_json = ?, "
            "difficulty = 0.2, search_context_json = ? "
            "WHERE id = ?",
            (
                source_video_id,
                start,
                end,
                clip_lessons[index],
                clip_lessons[index],
                json.dumps([clip_lessons[index]]),
                json.dumps(context),
                f"fresh-window-reel-{index}",
            ),
        )

    for index in range(8):
        source_video_id = f"parent-window-video-{index}"
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"fresh-window-reel-{index}",
            video_id=source_video_id,
            created_at=(now + timedelta(seconds=index)).isoformat(),
        )
        configure_window_reel(
            conn,
            index=index,
            source_video_id=source_video_id,
            chain_id=f"parent-chain-{index}",
            chain_position=0,
            prerequisite_ids=[],
        )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="fresh-organizer-candidate-window",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "slow",
            "num_reels": 9,
            "fresh_source_budget": True,
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="fresh-organizer-worker",
        now=now,
    )
    assert leased
    generated: list[dict] = []
    acquisition_calls: list[dict] = []
    organizer_calls: list[list[str]] = []
    selected_ids: list[str] = []

    def generate_stage(worker_conn, **kwargs) -> None:
        acquisition_calls.append(kwargs)
        kwargs["analyzed_video_ids"].update(
            {"fresh-source-one", "fresh-source-two", "fresh-source-three"}
        )
        for index in range(8, 128):
            fresh_index = index - 8
            generated.append(
                _insert_generation_reel(
                    worker_conn,
                    generation_id=str(kwargs["generation_id"]),
                    reel_id=f"fresh-window-reel-{index}",
                    video_id=f"fresh-window-video-{fresh_index}",
                    created_at=(now + timedelta(seconds=index)).isoformat(),
                )
            )
            configure_window_reel(
                worker_conn,
                index=index,
                source_video_id=f"fresh-window-video-{fresh_index % 3}",
                chain_id=f"source-chain-{fresh_index % 3}",
                chain_position=fresh_index // 3,
                prerequisite_ids=(
                    [] if fresh_index < 3 else [f"candidate-{index - 3}"]
                ),
            )

    def order_batch(reels, **kwargs):
        organizer_calls.append([reel["reel_id"] for reel in reels])
        assert set(organizer_calls[0]) == {
            f"fresh-window-reel-{index}" for index in range(128)
        }
        assert kwargs["release_limit"] == 128
        assert all(reel["concept_id"] == "c1" for reel in reels)
        assert all(
            reel["_selection_concept_family"]
            == "mitochondrial oxidative phosphorylation"
            for reel in reels
        )
        for reel in reels:
            index = int(str(reel["reel_id"]).rsplit("-", 1)[-1])
            assert reel["_selection_candidate_id"] == f"candidate-{index}"
            if index < 8:
                assert reel["_selection_chain_id"] == f"parent-chain-{index}"
                assert reel["_selection_chain_position"] == 0
                assert reel.get("_selection_prerequisite_ids", []) == []
            else:
                fresh_index = index - 8
                assert reel["_selection_chain_id"] == (
                    f"source-chain-{fresh_index % 3}"
                )
                assert reel["_selection_chain_position"] == fresh_index // 3
                assert reel.get("_selection_prerequisite_ids", []) == (
                    [] if fresh_index < 3 else [f"candidate-{index - 3}"]
                )
        assert all(reel["_selection_topic_relevance"] == 0.9 for reel in reels)
        assert all(reel["_selection_informativeness"] == 0.9 for reel in reels)
        assert {str(reel["ai_summary"]) for reel in reels} == set(clip_lessons)
        assert all(reel["takeaways"] for reel in reels)
        assert all(reel["transcript_snippet"] for reel in reels)
        selected = list(reels)
        selected_ids.extend(reel["reel_id"] for reel in selected)
        return mock.Mock(
            reels=selected,
            ordered_reel_ids=[reel["reel_id"] for reel in selected],
            assessment_checkpoint_reel_ids=[],
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert len(acquisition_calls) == 1
        assert acquisition_calls[0]["num_reels"] == 9
        assert acquisition_calls[0]["max_new_reels"] == 1
        assert acquisition_calls[0]["max_generation_videos"] == 3
        assert acquisition_calls[0]["retrieval_profile"] == "deep"
        assert acquisition_calls[0]["analyzed_video_ids"] == {
            "fresh-source-one",
            "fresh-source-two",
            "fresh-source-three",
        }
        assert len(organizer_calls) == 1
        assert main.LESSON_ORDER_CANDIDATE_LIMITS == {"fast": 88, "slow": 128}
        assert conn.execute(
            "SELECT COUNT(*) FROM reels WHERE generation_id = ?",
            (str(generation_jobs.get_job(conn, job["id"])["result_generation_id"]),),
        ).fetchone()[0] == 120
        final = next(
            event
            for event in generation_jobs.replay_events(conn, job_id=job["id"])
            if event["type"] == "final"
        )
        assert len(selected_ids) == 128
        assert [reel["reel_id"] for reel in final["payload"]["reels"]] == selected_ids
        assert all(
            not any(key.startswith("_selection_") for key in reel)
            for reel in final["payload"]["reels"]
        )
    finally:
        conn.close()


def test_raw_parent_inventory_prevents_top_up_after_smaller_selected_subset(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="raw-parent-inventory",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    for index in range(120):
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"raw-parent-reel-{index}",
            video_id=f"raw-parent-video-{index}",
            created_at=(now + timedelta(seconds=index)).isoformat(),
        )
        lesson = (
            "A distinct verified parent lesson about mitochondrial energy "
            f"with marker {index}."
        )
        context = json.loads(conn.execute(
            "SELECT search_context_json FROM reels WHERE id = ?",
            (f"raw-parent-reel-{index}",),
        ).fetchone()[0])
        context["selection_authority"] = "gemini"
        context["concept_family_contract_version"] = "concept_family_v3"
        context["concept_family"] = "mitochondrial oxidative phosphorylation"
        context["concept_aliases"] = []
        context["topic_evidence_quote"] = lesson
        conn.execute(
            "UPDATE reels SET transcript_snippet = ?, ai_summary = ?, "
            "takeaways_json = ?, search_context_json = ? WHERE id = ?",
            (
                lesson,
                lesson,
                json.dumps([lesson]),
                json.dumps(context),
                f"raw-parent-reel-{index}",
            ),
        )
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="raw-parent-release",
        generation_id=source_generation_id,
        completed_at=(now - timedelta(seconds=1)).isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "slow", "num_reels": 8},
    )
    _append_authoritative_release(
        conn,
        job_id=str(prior_job["id"]),
        reel_ids=[f"raw-parent-reel-{index}" for index in range(8)],
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=source_generation_id,
        metadata={
            "version": 2,
            "ordered_reel_ids": [
                f"raw-parent-reel-{index}" for index in range(8)
            ],
        },
    )
    request_params = {
        "generation_mode": "slow",
        "num_reels": 9,
        "fresh_source_budget": True,
    }
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="raw-parent-top-up",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params=request_params,
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="raw-parent-worker",
        now=now,
    )
    assert leased
    organizer_calls: list[list[str]] = []

    def order_batch(reels, **kwargs):
        organizer_calls.append([str(reel["reel_id"]) for reel in reels])
        assert set(organizer_calls[0]) == {
            f"raw-parent-reel-{index}" for index in range(8, 120)
        }
        assert kwargs["release_limit"] == 112
        selected = list(reels[:9])
        return mock.Mock(
            reels=selected,
            ordered_reel_ids=[reel["reel_id"] for reel in selected],
            assessment_checkpoint_reel_ids=[],
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    generate_reels = mock.Mock(
        side_effect=AssertionError("raw eligible inventory must prevent top-up")
    )
    monkeypatch.setattr(main.reel_service, "generate_reels", generate_reels)
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    try:
        assert len(
            main._reused_generation_reels(
                conn,
                generation_id=source_generation_id,
                material_id="m1",
                concept_id="c1",
                learner_id="learner-1",
                request_params=request_params,
                requested=9,
            )
        ) == 8
        assert main._current_level_reusable_generation_reel_count(
            conn,
            generation_id=source_generation_id,
            material_id="m1",
            concept_id="c1",
            learner_id="learner-1",
            request_params=request_params,
            requested=9,
        ) == main.INITIAL_READY_REEL_TARGET

        main._run_leased_generation_job(leased, threading.Event())

        generate_reels.assert_not_called()
        assert len(organizer_calls) == 1
        completed = generation_jobs.get_job(conn, job["id"])
        assert completed is not None
        assert completed["status"] == "completed"
    finally:
        conn.close()


def _patch_request_context(monkeypatch, conn: sqlite3.Connection) -> None:
    @contextmanager
    def connection(**_kwargs):
        yield conn

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "_require_community_client_identity", lambda _request: "owner")
    monkeypatch.setattr(
        main,
        "_require_verified_provider_account",
        lambda *_args, **_kwargs: {"id": "account-1"},
    )
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "_resolve_learner_identity", lambda *_args, **_kwargs: "learner-1")
    monkeypatch.setattr(
        main.reel_service,
        "learner_progress",
        lambda *_args, **_kwargs: {"selected_level": "beginner", "global_adjustment": 0.0},
    )


def _patch_transactional_worker_context(
    monkeypatch,
    conn: sqlite3.Connection,
) -> None:
    _patch_request_context(monkeypatch, conn)

    @contextmanager
    def connection(*, transactional: bool = False):
        if transactional:
            conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            if transactional:
                conn.rollback()
            raise
        else:
            if transactional:
                conn.commit()

    monkeypatch.setattr(main, "get_conn", connection)


def test_generate_submission_retries_serialization_failure_once(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    original_submit = main._submit_bounded_generation_job
    submit_calls = 0

    def flaky_submit(worker_conn, **kwargs):
        nonlocal submit_calls
        submit_calls += 1
        if submit_calls == 1:
            raise _PostgresTransactionFailure("40001")
        return original_submit(worker_conn, **kwargs)

    monkeypatch.setattr(main, "_submit_bounded_generation_job", flaky_submit)
    try:
        response = asyncio.run(
            main.generate_reels(object(), ReelsGenerateRequest(material_id="m1"))
        )
        assert response.status_code == 202
        assert submit_calls == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_generate_submission_lost_commit_ack_converges_on_one_job(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    attempts = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal attempts
        assert transactional is True
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
            attempts += 1
            if attempts == 1:
                raise _PostgresTransactionFailure("08006")

    monkeypatch.setattr(main, "get_conn", connection)
    try:
        response = asyncio.run(
            main.generate_reels(object(), ReelsGenerateRequest(material_id="m1"))
        )
        payload = json.loads(response.body)
        assert response.status_code == 202
        assert attempts == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 1
        assert generation_jobs.get_job(conn, payload["job_id"])["status"] == "queued"
        wake.assert_called_once_with()
    finally:
        conn.close()


def test_feed_ranking_retries_serialization_failure_on_fresh_checkout(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    attempts = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal attempts
        assert transactional is True
        attempts += 1
        if attempts == 1:
            raise _PostgresTransactionFailure("40001")
        yield conn

    monkeypatch.setattr(main, "get_conn", connection)
    try:
        response = main.feed(object(), material_id="m1", autofill=False)
        assert response["reels"] == []
        assert attempts == 2
    finally:
        conn.close()


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
                "selection_contract_version": "quality_silence_v39",
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
        "selection_contract_version": "quality_silence_v39",
    }


def test_generation_worker_retries_setup_and_release_transactions_without_replaying_work(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="setup-release-db-retry",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="setup-release-retry-worker",
        now=now,
    )
    assert leased
    generated: list[dict] = []
    retrieval_calls = 0
    setup_attempts = 0
    release_attempts = 0
    original_progress = main.update_generation_progress
    original_terminal = main.transition_generation_terminal

    def generate_stage(worker_conn, **kwargs) -> None:
        nonlocal retrieval_calls
        retrieval_calls += 1
        generated.append(
            _insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id="retry-release-reel",
                video_id="retry-release-video",
                created_at=now.isoformat(),
            )
        )

    def transient_setup(worker_conn, **kwargs):
        nonlocal setup_attempts
        result = original_progress(worker_conn, **kwargs)
        if kwargs.get("phase") == "retrieval":
            setup_attempts += 1
            if setup_attempts == 1:
                raise _PostgresTransactionFailure("40001")
        return result

    def transient_release(worker_conn, **kwargs):
        nonlocal release_attempts
        if kwargs.get("status") == "completed":
            release_attempts += 1
            if release_attempts == 1:
                raise _PostgresTransactionFailure("40P01")
        return original_terminal(worker_conn, **kwargs)

    def order_batch(reels, **_kwargs):
        return mock.Mock(
            reels=list(reels),
            ordered_reel_ids=[reel["reel_id"] for reel in reels],
            assessment_checkpoint_reel_ids=[],
            model_used=None,
            degraded=False,
            fallback_reason=None,
            provider_called=False,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(generated),
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    monkeypatch.setattr(main, "update_generation_progress", transient_setup)
    monkeypatch.setattr(main, "transition_generation_terminal", transient_release)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed = generation_jobs.get_job(conn, str(job["id"]))
        assert completed is not None
        assert completed["status"] == "completed"
        assert setup_attempts == 2
        assert release_attempts == 2
        assert retrieval_calls == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generations WHERE request_key = ?",
            ("setup-release-db-retry",),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_heads WHERE active_generation_id = ?",
            (completed["result_generation_id"],),
        ).fetchone()[0] == 1
        assert [
            event["type"]
            for event in generation_jobs.replay_events(conn, job_id=str(job["id"]))
        ] == ["final", "terminal"]
    finally:
        conn.close()


def test_generation_worker_setup_lost_commit_ack_converges_before_provider_work(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="setup-lost-ack-retry",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="setup-lost-ack-worker",
        now=now,
    )
    assert leased
    transaction_commits = 0
    first_committed_generation_id = ""
    retrieval_calls = 0
    created_generation_ids: list[str] = []
    original_create_generation = main._create_generation_row

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal transaction_commits, first_committed_generation_id
        if transactional:
            conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            if transactional:
                conn.rollback()
            raise
        else:
            if transactional:
                conn.commit()
                transaction_commits += 1
                if transaction_commits == 1:
                    committed = generation_jobs.get_job(conn, str(job["id"])) or {}
                    first_committed_generation_id = str(
                        committed.get("result_generation_id") or ""
                    )
                    raise _PostgresTransactionFailure("08006")

    def track_generation_create(worker_conn, **kwargs):
        created_id = original_create_generation(worker_conn, **kwargs)
        created_generation_ids.append(created_id)
        return created_id

    def fail_retrieval(_worker_conn, **_kwargs) -> None:
        nonlocal retrieval_calls
        retrieval_calls += 1
        raise ProviderTransientError(
            "stop after setup",
            provider="supadata",
            operation="search",
        )

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "_create_generation_row", track_generation_create)
    monkeypatch.setattr(main.reel_service, "generate_reels", fail_retrieval)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        requeued = generation_jobs.get_job(conn, str(job["id"]))
        assert requeued is not None
        assert requeued["status"] == "queued"
        assert requeued["terminal_error_code"] is None
        assert retrieval_calls == 1
        assert first_committed_generation_id
        assert requeued["result_generation_id"] == first_committed_generation_id
        assert created_generation_ids == [first_committed_generation_id]
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generations WHERE request_key = ?",
            ("setup-lost-ack-retry",),
        ).fetchone()[0] == 1
        assert generation_jobs.replay_events(conn, job_id=str(job["id"])) == []
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (RuntimeError("unexpected generation failure"), "generation_failed"),
    ],
)
def test_generation_worker_retries_failure_terminalization_once(
    monkeypatch,
    failure,
    expected_code,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"terminal-db-retry-{expected_code}",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner=f"terminal-retry-{expected_code}",
        now=now,
    )
    assert leased
    retrieval_calls = 0
    terminal_attempts = 0
    original_terminal = main.transition_generation_terminal

    def fail_retrieval(_worker_conn, **_kwargs) -> None:
        nonlocal retrieval_calls
        retrieval_calls += 1
        raise failure

    def transient_terminal(worker_conn, **kwargs):
        nonlocal terminal_attempts
        terminal_attempts += 1
        if terminal_attempts == 1:
            raise _PostgresTransactionFailure("40001")
        return original_terminal(worker_conn, **kwargs)

    monkeypatch.setattr(main.reel_service, "generate_reels", fail_retrieval)
    monkeypatch.setattr(main, "transition_generation_terminal", transient_terminal)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        failed = generation_jobs.get_job(conn, str(job["id"]))
        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["terminal_error_code"] == expected_code
        assert retrieval_calls == 1
        assert terminal_attempts == 2
        assert [
            event["type"]
            for event in generation_jobs.replay_events(conn, job_id=str(job["id"]))
        ] == ["terminal"]
    finally:
        conn.close()


def test_generation_worker_retries_transient_provider_twice_then_succeeds_same_job(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-retry-success-third-attempt",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    generated: list[dict] = []
    retrieval_calls: list[dict] = []
    wake = mock.Mock()
    ledger_recovery = mock.Mock(wraps=main._generation_gemini_ledger_exposure)

    def generate_stage(worker_conn, **kwargs) -> None:
        retrieval_calls.append(kwargs)
        reservation = kwargs["generation_context"].reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=100,
            max_output_tokens=100,
            max_physical_attempts=1,
        )
        kwargs["generation_context"].record_gemini(
            operation="segmentation",
            attempt=1,
            model_used="gemini-3.1-pro-preview",
            quality_degraded=False,
            usage={
                **reservation,
                "prompt_tokens": 10,
                "candidate_tokens": 2,
                "total_tokens": 12,
            },
            status_code=503 if len(retrieval_calls) <= 2 else 200,
            error_code=(
                "provider_transient" if len(retrieval_calls) <= 2 else ""
            ),
            stage="segmentation",
        )
        if len(retrieval_calls) <= 2:
            assert "AAAAAAAAAAA" not in kwargs["exclude_video_ids"]
            kwargs["retrieved_video_ids"].add("AAAAAAAAAAA")
            kwargs["attempted_video_ids"].add("AAAAAAAAAAA")
            kwargs["generation_context"].increment_counter("provider_cursor_open")
            raise ProviderTransientError(
                "Gemini is temporarily unavailable.",
                provider="gemini",
                operation="segmentation",
                status_code=503,
            )
        assert "AAAAAAAAAAA" in kwargs["exclude_video_ids"]
        kwargs["retrieved_video_ids"].add("BBBBBBBBBBB")
        kwargs["attempted_video_ids"].add("BBBBBBBBBBB")
        kwargs["analyzed_video_ids"].add("BBBBBBBBBBB")
        generated.append(
            _insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id="provider-retry-reel",
                video_id="BBBBBBBBBBB",
                created_at=now.isoformat(),
            )
        )

    def order_batch(reels, **_kwargs):
        return mock.Mock(
            reels=list(reels),
            ordered_reel_ids=[reel["reel_id"] for reel in reels],
            assessment_checkpoint_reel_ids=[],
            model_used=None,
            degraded=False,
            fallback_reason=None,
            provider_called=False,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(generated),
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    monkeypatch.setattr(
        main,
        "_generation_gemini_ledger_exposure",
        ledger_recovery,
    )
    try:
        generation_id = ""
        for attempt in (1, 2, 3):
            leased = generation_jobs.lease_job(
                conn,
                job_id=str(job["id"]),
                lease_owner=f"provider-retry-worker-{attempt}",
                now=now + timedelta(seconds=attempt - 1),
            )
            assert leased and leased["attempt_count"] == attempt
            main._run_leased_generation_job(leased, threading.Event())
            current = generation_jobs.get_job(conn, str(job["id"]))
            assert current is not None
            if generation_id:
                assert current["result_generation_id"] == generation_id
            else:
                generation_id = str(current["result_generation_id"] or "")
                assert generation_id
            if attempt < 3:
                assert current["status"] == "queued"
                usage = json.loads(str(current["usage_json"] or "{}"))
                assert usage["failed_source_attempts"] == {
                    "AAAAAAAAAAA": attempt,
                }
                assert len(usage["retry_errors"]) == attempt
                assert generation_jobs.replay_events(
                    conn,
                    job_id=str(job["id"]),
                ) == []
            else:
                assert current["status"] == "completed"

        completed = generation_jobs.get_job(conn, str(job["id"]))
        assert completed and completed["attempt_count"] == 3
        usage = json.loads(str(completed["usage_json"] or "{}"))
        assert usage["consumed_video_ids"] == ["BBBBBBBBBBB"]
        assert usage["failed_source_attempts"] == {"AAAAAAAAAAA": 2}
        assert len(usage["retry_errors"]) == 2
        assert usage["counters"]["durable_attempts"] == 3
        assert usage["counters"]["provider_cursor_open"] == 0
        assert len(usage["provider_calls"]) == 3
        assert usage["summary"]["gemini_calls"] == 3
        assert usage["summary"]["input_tokens"] == 30
        assert usage["by_stage"]["segmentation"]["calls"] == 3
        assert len(usage["attempt_budgets"]) == 2
        attempt_exposures = [
            attempt_budget["gemini"]["cost_exposure_usd"]
            for attempt_budget in usage["attempt_budgets"]
        ]
        assert 0 < attempt_exposures[0] < attempt_exposures[1]
        final_budget = usage["budget"]["gemini"]
        assert attempt_exposures[1] < final_budget["cost_exposure_usd"]
        assert final_budget["cost_exposure_usd"] <= final_budget["cost_limit_usd"]
        assert final_budget["selector_calls"] == 1
        assert usage["summary"]["lifetime_reserved_worst_case_cost_usd"] == (
            final_budget["lifetime_reserved_worst_case_cost_usd"]
        )
        assert main._generation_job_has_retryable_source_work(conn, completed) is False
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generations WHERE request_key = ?",
            ("provider-retry-success-third-attempt",),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reels WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM generation_provider_usage WHERE job_id = ?",
            (str(job["id"]),),
        ).fetchone()[0] == 3
        assert [
            event["type"]
            for event in generation_jobs.replay_events(conn, job_id=str(job["id"]))
        ] == ["final", "terminal"]
        assert wake.call_count == 2
        assert len(retrieval_calls) == 3
        assert ledger_recovery.call_count == 2
    finally:
        conn.close()


def test_same_job_retry_composes_with_ancestral_source_attempt_bound(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="ancestral-source-failure",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    source_job = _terminal_job_for_generation(
        conn,
        request_key="ancestral-source-failure",
        generation_id=source_generation_id,
        completed_at=now.isoformat(),
        status="failed",
    )
    conn.execute(
        "UPDATE reel_generation_jobs SET usage_json = ? WHERE id = ?",
        (
            json.dumps({
                "counters": {"provider_cursor_open": 0},
                "consumed_video_ids": [],
                "failed_source_attempts": {"AAAAAAAAAAA": 1},
            }),
            source_job["id"],
        ),
    )
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="same-job-with-ancestral-source-failure",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        source_generation_id=source_generation_id,
        now=now,
    )
    generated: list[dict] = []
    retrieval_calls = 0

    def generate_stage(worker_conn, **kwargs) -> None:
        nonlocal retrieval_calls
        retrieval_calls += 1
        if retrieval_calls == 1:
            assert "AAAAAAAAAAA" not in kwargs["exclude_video_ids"]
            kwargs["retrieved_video_ids"].add("AAAAAAAAAAA")
            kwargs["attempted_video_ids"].add("AAAAAAAAAAA")
            raise ProviderTransientError(
                "Gemini is temporarily unavailable.",
                provider="gemini",
                operation="segmentation",
                status_code=503,
            )
        assert "AAAAAAAAAAA" in kwargs["exclude_video_ids"]
        kwargs["retrieved_video_ids"].add("BBBBBBBBBBB")
        kwargs["attempted_video_ids"].add("BBBBBBBBBBB")
        kwargs["analyzed_video_ids"].add("BBBBBBBBBBB")
        generated.append(
            _insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id="ancestral-retry-reel",
                video_id="BBBBBBBBBBB",
                created_at=now.isoformat(),
            )
        )

    def order_batch(reels, **_kwargs):
        return mock.Mock(
            reels=list(reels),
            ordered_reel_ids=[reel["reel_id"] for reel in reels],
            assessment_checkpoint_reel_ids=[],
            model_used=None,
            degraded=False,
            fallback_reason=None,
            provider_called=False,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(generated),
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    try:
        first = generation_jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="ancestral-retry-worker-1",
            now=now,
        )
        assert first
        main._run_leased_generation_job(first, threading.Event())
        requeued = generation_jobs.get_job(conn, job["id"])
        assert requeued and requeued["status"] == "queued"
        assert json.loads(requeued["usage_json"])["failed_source_attempts"] == {
            "AAAAAAAAAAA": 1,
        }

        second = generation_jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="ancestral-retry-worker-2",
            now=now + timedelta(seconds=1),
        )
        assert second and second["attempt_count"] == 2
        main._run_leased_generation_job(second, threading.Event())

        completed = generation_jobs.get_job(conn, job["id"])
        assert completed and completed["status"] == "completed"
        current_usage = json.loads(completed["usage_json"])
        assert current_usage["failed_source_attempts"] == {"AAAAAAAAAAA": 1}
        assert main._generation_chain_failed_source_attempts(
            conn,
            generation_id=str(completed["result_generation_id"]),
        ) == {"AAAAAAAAAAA": 2}
        assert retrieval_calls == 2
    finally:
        conn.close()


def test_generation_worker_terminalizes_once_after_three_transient_failures(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-retry-exhausted-third-attempt",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    retrieval_calls = 0

    def fail_stage(_worker_conn, **kwargs) -> None:
        nonlocal retrieval_calls
        retrieval_calls += 1
        if retrieval_calls <= 2:
            kwargs["retrieved_video_ids"].add("AAAAAAAAAAA")
            kwargs["attempted_video_ids"].add("AAAAAAAAAAA")
        else:
            assert "AAAAAAAAAAA" in kwargs["exclude_video_ids"]
        raise ProviderTransientError(
            "Provider is temporarily unavailable.",
            provider="gemini" if retrieval_calls <= 2 else "supadata",
            operation="segmentation" if retrieval_calls <= 2 else "search",
            status_code=503,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", fail_stage)
    try:
        for attempt in (1, 2, 3):
            leased = generation_jobs.lease_job(
                conn,
                job_id=str(job["id"]),
                lease_owner=f"provider-failure-worker-{attempt}",
                now=now + timedelta(seconds=attempt - 1),
            )
            assert leased and leased["attempt_count"] == attempt
            main._run_leased_generation_job(leased, threading.Event())

        failed = generation_jobs.get_job(conn, str(job["id"]))
        assert failed and failed["status"] == "failed"
        assert failed["attempt_count"] == failed["max_attempts"] == 3
        assert failed["terminal_error_code"] == "provider_transient"
        usage = json.loads(str(failed["usage_json"] or "{}"))
        assert usage["failed_source_attempts"] == {"AAAAAAAAAAA": 2}
        assert len(usage["retry_errors"]) == 3
        assert usage["counters"]["provider_cursor_open"] == 0
        assert main._generation_job_has_retryable_source_work(conn, failed) is False
        assert [
            event["type"]
            for event in generation_jobs.replay_events(conn, job_id=str(job["id"]))
        ] == ["terminal"]
        assert generation_jobs.lease_job(
            conn,
            job_id=str(job["id"]),
            lease_owner="provider-failure-worker-4",
            now=now + timedelta(seconds=3),
        ) is None
        assert retrieval_calls == 3
    finally:
        conn.close()


def test_generation_worker_retries_response_validation_three_times_with_precise_error(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-response-validation-retry-exhausted",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    attempts = 0
    detail = (
        "GeminiSelectorContractError:"
        "intent_contract_incomplete_joint_structure"
    )

    def fail_stage(_worker_conn, **_kwargs) -> None:
        nonlocal attempts
        attempts += 1
        raise ProviderResponseValidationError(
            "Gemini responded, but its clip plan did not satisfy "
            "the learning-request contract.",
            provider="gemini",
            operation="segmentation",
            status_code=200,
            detail=detail,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", fail_stage)
    try:
        for attempt in (1, 2, 3):
            leased = generation_jobs.lease_job(
                conn,
                job_id=str(job["id"]),
                lease_owner=f"response-validation-worker-{attempt}",
                now=now + timedelta(seconds=attempt - 1),
            )
            assert leased and leased["attempt_count"] == attempt
            main._run_leased_generation_job(leased, threading.Event())

            current = generation_jobs.get_job(conn, str(job["id"]))
            assert current is not None
            assert current["status"] == (
                "queued" if attempt < 3 else "failed"
            )
            usage = json.loads(str(current["usage_json"] or "{}"))
            assert len(usage["retry_errors"]) == attempt
            assert usage["retry_errors"][-1] == {
                "code": "provider_response_invalid",
                "message": (
                    "Gemini responded, but its clip plan did not satisfy "
                    "the learning-request contract."
                ),
                "provider": "gemini",
                "operation": "segmentation",
                "retryable": True,
                "status_code": 200,
                "detail": detail,
            }

        failed = generation_jobs.get_job(conn, str(job["id"]))
        assert failed is not None
        assert failed["attempt_count"] == failed["max_attempts"] == 3
        assert failed["terminal_error_code"] == "provider_response_invalid"
        assert "unavailable" not in str(
            failed["terminal_error_message"]
        ).casefold()
        status = main._generation_job_status_payload(conn, failed)
        assert status["error"]["code"] == "provider_response_invalid"
        assert status["error"]["detail"]["detail"] == detail
        assert status["error"]["detail"]["status_code"] == 200
        assert attempts == 3
    finally:
        conn.close()


def test_reclaimed_lease_restores_gemini_exposure_from_durable_usage_ledger(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-ledger-budget-recovery",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    first = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="provider-ledger-crashed-worker",
        lease_seconds=1,
        now=now,
    )
    assert first and first["attempt_count"] == 1
    generation_jobs.record_provider_usage(
        conn,
        job_id=str(job["id"]),
        provider="gemini",
        operation="segmentation",
        model="gemini-3.1-pro-preview",
        billable_requests=1,
        input_tokens=100_000,
        output_tokens=60_000,
        total_tokens=160_000,
        metadata={
            "provider_call": True,
            "billing_usage_known": True,
            "cached_tokens": 0,
            "reserved_cost_usd": 0.95,
            "admitted_cost_usd": 0.95,
        },
        now=now,
    )
    reclaimed = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="provider-ledger-recovery-worker",
        now=now + timedelta(seconds=2),
    )
    assert reclaimed and reclaimed["attempt_count"] == 2
    seen_exposure: list[float] = []

    def generate_stage(_worker_conn, **kwargs) -> None:
        budget = kwargs["generation_context"].budget
        seen_exposure.append(budget.snapshot()["gemini"]["cost_exposure_usd"])
        budget_context = kwargs["generation_context"]
        budget_context.reserve_gemini_call(
            operation="pro_authoritative",
            model="gemini-3.1-pro-preview",
            estimated_input_tokens=100_000,
            max_output_tokens=6_000,
            max_physical_attempts=1,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    try:
        main._run_leased_generation_job(reclaimed, threading.Event())

        failed = generation_jobs.get_job(conn, str(job["id"]))
        assert failed and failed["status"] == "failed"
        assert failed["attempt_count"] == 2
        assert failed["terminal_error_code"] == "provider_budget_exceeded"
        assert seen_exposure == [pytest.approx(0.92)]
        usage = json.loads(str(failed["usage_json"] or "{}"))
        gemini_budget = usage["budget"]["gemini"]
        assert gemini_budget["cost_exposure_usd"] == pytest.approx(0.92)
        assert gemini_budget["cost_exposure_usd"] <= gemini_budget["cost_limit_usd"]
        assert gemini_budget["lifetime_reserved_worst_case_cost_usd"] == pytest.approx(
            0.95
        )
    finally:
        conn.close()


def test_retry_ledger_read_failure_never_starts_or_renews_heartbeat(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-ledger-read-failure",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    first = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="provider-ledger-read-crashed-worker",
        lease_seconds=1,
        now=now,
    )
    assert first
    reclaimed = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="provider-ledger-read-worker-2",
        lease_seconds=1,
        now=now + timedelta(seconds=2),
    )
    assert reclaimed and reclaimed["attempt_count"] == 2
    heartbeat_thread = mock.Mock()
    monkeypatch.setattr(
        main.threading,
        "Thread",
        mock.Mock(return_value=heartbeat_thread),
    )
    monkeypatch.setattr(
        main,
        "_generation_gemini_ledger_exposure",
        mock.Mock(side_effect=RuntimeError("provider ledger unavailable")),
    )
    try:
        with pytest.raises(RuntimeError, match="provider ledger unavailable"):
            main._run_leased_generation_job(reclaimed, threading.Event())

        heartbeat_thread.start.assert_not_called()
        heartbeat_thread.join.assert_not_called()
        third = generation_jobs.lease_job(
            conn,
            job_id=str(job["id"]),
            lease_owner="provider-ledger-read-worker-3",
            lease_seconds=1,
            now=now + timedelta(seconds=4),
        )
        assert third and third["attempt_count"] == 3
    finally:
        conn.close()


def test_generation_worker_requeue_converges_after_lost_commit_ack(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-requeue-lost-ack",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="provider-requeue-lost-ack-worker",
        now=now,
    )
    assert leased
    ack_lost = False
    requeue_calls = 0
    original_requeue = main.requeue_generation_retryable_failure

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal ack_lost
        if transactional:
            conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            if transactional:
                conn.rollback()
            raise
        else:
            if transactional:
                conn.commit()
                current = generation_jobs.get_job(conn, str(job["id"])) or {}
                if current.get("status") == "queued" and not ack_lost:
                    ack_lost = True
                    raise _PostgresTransactionFailure("08006")

    def track_requeue(worker_conn, **kwargs):
        nonlocal requeue_calls
        requeue_calls += 1
        return original_requeue(worker_conn, **kwargs)

    def fail_stage(_worker_conn, **_kwargs) -> None:
        raise ProviderTransientError(
            "Gemini is temporarily unavailable.",
            provider="gemini",
            operation="segmentation",
            status_code=503,
        )

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(main, "requeue_generation_retryable_failure", track_requeue)
    monkeypatch.setattr(main.reel_service, "generate_reels", fail_stage)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        requeued = generation_jobs.get_job(conn, str(job["id"]))
        assert ack_lost is True
        assert requeue_calls == 2
        assert requeued and requeued["status"] == "queued"
        assert requeued["attempt_count"] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generations WHERE request_key = ?",
            ("provider-requeue-lost-ack",),
        ).fetchone()[0] == 1
        assert generation_jobs.replay_events(conn, job_id=str(job["id"])) == []
    finally:
        conn.close()


def test_generation_worker_retries_lesson_and_recall_writes_without_rerunning_organizer(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _created = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="lesson-recall-db-retry",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=str(job["id"]),
        lease_owner="lesson-recall-retry-worker",
        now=now,
    )
    assert leased
    generated: list[dict] = []
    retrieval_calls = 0
    organizer_calls = 0
    lesson_write_attempts = 0
    recall_attempts = 0
    original_lesson_persist = main._persist_generation_lesson_order
    original_prepare = main.assessment_service.prepare_reel_questions

    def generate_stage(worker_conn, **kwargs) -> None:
        nonlocal retrieval_calls
        retrieval_calls += 1
        generated.append(
            _insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id="retry-checkpoint-reel",
                video_id="retry-checkpoint-video",
                created_at=now.isoformat(),
            )
        )

    def order_batch(reels, **_kwargs):
        nonlocal organizer_calls
        organizer_calls += 1
        return mock.Mock(
            reels=list(reels),
            ordered_reel_ids=["retry-checkpoint-reel"],
            assessment_checkpoint_reel_ids=["retry-checkpoint-reel"],
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    def transient_lesson_write(worker_conn, **kwargs):
        nonlocal lesson_write_attempts
        lesson_write_attempts += 1
        original_lesson_persist(worker_conn, **kwargs)
        if lesson_write_attempts == 1:
            raise _PostgresTransactionFailure("40001")

    def transient_recall(worker_conn, **kwargs):
        nonlocal recall_attempts
        recall_attempts += 1
        result = original_prepare(worker_conn, **kwargs)
        if recall_attempts == 1:
            raise _PostgresTransactionFailure("40P01")
        return result

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(generated),
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    monkeypatch.setattr(
        main,
        "_persist_generation_lesson_order",
        transient_lesson_write,
    )
    monkeypatch.setattr(
        main.assessment_service,
        "prepare_reel_questions",
        transient_recall,
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed = generation_jobs.get_job(conn, str(job["id"]))
        assert completed is not None
        assert completed["status"] == "completed"
        assert retrieval_calls == 1
        assert organizer_calls == 1
        assert lesson_write_attempts == 2
        assert recall_attempts == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_assessment_questions "
            "WHERE reel_id = 'retry-checkpoint-reel'"
        ).fetchone()[0] == 1
        generation_row = main._fetch_generation_row(
            conn,
            str(completed["result_generation_id"]),
        )
        metadata = json.loads(str(generation_row["lesson_order_json"]))
        assert metadata["assessment_checkpoint_reel_ids"] == [
            "retry-checkpoint-reel"
        ]
        assert metadata["degraded"] is False
    finally:
        conn.close()


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
                "selection_contract_version": "quality_silence_v39",
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


def test_latest_compatible_job_requires_current_adaptation_fingerprint() -> None:
    conn = _conn()
    params = {
        "request_schema_version": main.GENERATION_REQUEST_SCHEMA_VERSION,
        "generation_mode": "slow",
        "creative_commons_only": False,
        "preferred_video_duration": "any",
        "knowledge_level": "beginner",
        "language": "en",
        "exclude_video_ids": [],
        "min_relevance": None,
        "adaptation_fingerprint": "before-feedback",
    }
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="before-feedback",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    content_fingerprint = generation_jobs.material_content_fingerprint(
        conn, "m1", None
    )
    prior = _terminal_job_for_generation(
        conn,
        request_key="before-feedback",
        generation_id=generation_id,
        completed_at="2026-07-12T00:00:00+00:00",
        concept_id=None,
        content_fingerprint=content_fingerprint,
        request_params=params,
    )
    try:
        matching = main._latest_compatible_generation_job(
            conn,
            material_id="m1",
            learner_id="learner-1",
            concept_id=None,
            content_fingerprint=content_fingerprint,
            request_params=params,
        )
        changed = main._latest_compatible_generation_job(
            conn,
            material_id="m1",
            learner_id="learner-1",
            concept_id=None,
            content_fingerprint=content_fingerprint,
            request_params={**params, "adaptation_fingerprint": "after-feedback"},
        )

        assert matching is not None
        assert matching["id"] == prior["id"]
        assert changed is None
    finally:
        conn.close()


def _append_authoritative_release(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    reel_ids: list[str],
) -> None:
    generation_jobs.append_event(
        conn,
        job_id=job_id,
        event_type="final",
        payload={
            "reels": [{"reel_id": reel_id} for reel_id in reel_ids],
            "authoritative": True,
        },
    )


def _released_feed_reel(reel_id: str, index: int) -> dict:
    return {
        "reel_id": reel_id,
        "video_id": f"video-{reel_id}",
        "t_start": float(index * 40),
        "t_end": float(index * 40 + 30),
        "difficulty": 0.1,
        "_selection_quality_floor": 0.9,
        "_selection_quality_mean": 0.9,
        "_selection_topic_relevance": 0.9,
        "_selection_source_rank": index,
        "_selection_ordered": True,
        "selection_contract_version": "quality_silence_v39",
    }


def _persist_released_feed_reel(
    conn: sqlite3.Connection,
    *,
    generation_id: str,
    reel: dict,
) -> None:
    video_id = str(reel["video_id"])
    conn.execute(
        "INSERT OR IGNORE INTO videos "
        "(id, title, channel_title, duration_sec, created_at) "
        "VALUES (?, ?, 'Test', 300, '2026-07-20T00:00:00+00:00')",
        (video_id, video_id),
    )
    conn.execute(
        "INSERT INTO reels "
        "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
        "transcript_snippet, takeaways_json, base_score, difficulty, generation_id, "
        "search_context_json, created_at) "
        "VALUES (?, 'm1', 'c1', ?, '', ?, ?, '', '[]', 1, 0.1, ?, ?, "
        "'2026-07-20T00:00:00+00:00')",
        (
            str(reel["reel_id"]),
            video_id,
            float(reel["t_start"]),
            float(reel["t_end"]),
            generation_id,
            json.dumps(reel.get("search_context") or {}),
        ),
    )


def _released_generation_chain(
    conn: sqlite3.Connection,
    batches: list[tuple[list[str], list[str]]],
) -> tuple[str, dict[str, list[dict]], list[dict]]:
    source_generation_id: str | None = None
    prior_job_id: str | None = None
    rows_by_generation: dict[str, list[dict]] = {}
    jobs: list[dict] = []
    row_index = 0
    completed_at = datetime.now(timezone.utc)
    for batch_index, (released_ids, raw_ids) in enumerate(batches):
        generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=f"released-generation-{batch_index}",
            generation_mode="slow",
            retrieval_profile="unified",
            source_generation_id=source_generation_id,
        )
        request_params = {"generation_mode": "slow", "num_reels": 20}
        if prior_job_id:
            request_params["continuation_token"] = prior_job_id
        job = _terminal_job_for_generation(
            conn,
            request_key=f"released-job-{batch_index}",
            generation_id=generation_id,
            completed_at=(completed_at + timedelta(seconds=batch_index)).isoformat(),
            request_params=request_params,
        )
        _append_authoritative_release(
            conn,
            job_id=str(job["id"]),
            reel_ids=released_ids,
        )
        rows_by_generation[generation_id] = [
            _released_feed_reel(reel_id, row_index + index)
            for index, reel_id in enumerate(raw_ids)
        ]
        row_index += len(raw_ids)
        source_generation_id = generation_id
        prior_job_id = str(job["id"])
        jobs.append(job)
    assert source_generation_id is not None
    return source_generation_id, rows_by_generation, jobs


def _patch_released_ranked_feed(
    monkeypatch,
    rows_by_generation: dict[str, list[dict]],
) -> None:
    monkeypatch.setattr(
        main.reel_service,
        "ranked_feed",
        lambda *_args, **kwargs: list(rows_by_generation[kwargs["generation_id"]]),
    )
    monkeypatch.setattr(
        main.reel_service,
        "learner_progress",
        lambda *_args, **_kwargs: {"selected_level": "beginner"},
    )


def _rank_released_feed(
    conn: sqlite3.Connection,
    *,
    generation_id: str,
    exclude_reel_ids: list[str] | None = None,
    exclude_video_ids: list[str] | None = None,
    released_only: bool = True,
) -> list[dict]:
    return main._ranked_request_reels(
        conn,
        material_id="m1",
        fast_mode=False,
        generation_id=generation_id,
        min_relevance=None,
        preferred_video_duration="any",
        target_clip_duration_sec=0,
        target_clip_duration_min_sec=None,
        target_clip_duration_max_sec=None,
        exclude_video_ids=exclude_video_ids,
        exclude_reel_ids=exclude_reel_ids,
        page=1,
        limit=20,
        released_only=released_only,
    )


def test_released_feed_reconstructs_exact_batch_order_and_hides_raw_reservoir(
    monkeypatch,
) -> None:
    conn = _conn()
    latest_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["r1", "r2", "r3", "r4"], ["r1", "r2", "r3", "r4"]),
            (["r5", "r6", "r7"], ["r5", "r6", "r7"]),
            (["r8", "r9"], ["r8", "r9", "raw-private"]),
        ],
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(
            conn,
            generation_id=latest_generation_id,
        )

        assert [reel["reel_id"] for reel in ranked] == [
            "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9",
        ]
    finally:
        conn.close()


def test_terminal_summary_marker_projects_to_first_surviving_recap() -> None:
    assert main._surviving_terminal_summary_start_reel_id(
        ordered_reel_ids=["teaching", "removed-recap", "surviving-recap"],
        terminal_summary_start_reel_id="removed-recap",
        surviving_reel_ids=["teaching", "surviving-recap"],
    ) == "surviving-recap"
    assert main._surviving_terminal_summary_start_reel_id(
        ordered_reel_ids=["teaching", "removed-recap"],
        terminal_summary_start_reel_id="removed-recap",
        surviving_reel_ids=["teaching"],
    ) is None


def test_authoritative_chain_places_later_teaching_before_earlier_recap(
    monkeypatch,
) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["first-teaching", "first-recap"], ["first-teaching", "first-recap"]),
            (["later-teaching", "later-recap"], ["later-teaching", "later-recap"]),
        ],
    )
    source_generation_id = next(iter(rows))
    try:
        main._persist_generation_lesson_order(
            conn,
            generation_id=source_generation_id,
            metadata={
                "version": 2,
                "ordered_reel_ids": ["first-teaching", "first-recap"],
                "terminal_summary_start_reel_id": "first-recap",
            },
        )
        main._persist_generation_lesson_order(
            conn,
            generation_id=child_generation_id,
            metadata={
                "version": 2,
                "ordered_reel_ids": ["later-teaching", "later-recap"],
                "terminal_summary_start_reel_id": "later-recap",
            },
        )

        assert main._authoritative_release_reel_ids(
            conn,
            child_generation_id,
        ) == [
            "first-teaching",
            "later-teaching",
            "first-recap",
            "later-recap",
        ]
        _patch_released_ranked_feed(monkeypatch, rows)
        ranked = _rank_released_feed(conn, generation_id=child_generation_id)
        assert [reel["reel_id"] for reel in ranked] == [
            "first-teaching",
            "later-teaching",
            "first-recap",
            "later-recap",
        ]
    finally:
        conn.close()


def test_released_feed_drops_child_span_containing_prior_release(monkeypatch) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["initial-clip"], ["initial-clip"]),
            (["child-overlap"], ["child-overlap"]),
        ],
    )
    source_generation_id = next(iter(rows))
    initial = rows[source_generation_id][0]
    initial.update(video_id="LQyFshgm-hU", t_start=38.5, t_end=60.4)
    child = rows[child_generation_id][0]
    child.update(video_id="LQyFshgm-hU", t_start=38.5, t_end=113.3)
    _persist_released_feed_reel(
        conn,
        generation_id=source_generation_id,
        reel=initial,
    )
    _persist_released_feed_reel(
        conn,
        generation_id=child_generation_id,
        reel=child,
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=source_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["initial-clip"]},
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=child_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["child-overlap"]},
    )
    assert main._filter_continuation_release_temporal_overlaps(
        conn,
        source_generation_id=source_generation_id,
        generation_id=child_generation_id,
        reels=[child],
    ) == []
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(conn, generation_id=child_generation_id)

        assert [reel["reel_id"] for reel in ranked] == ["initial-clip"]
        monkeypatch.setattr(
            main,
            "_learner_seen_reel_ids",
            lambda *_args, **_kwargs: {"initial-clip"},
        )
        assert _rank_released_feed(conn, generation_id=child_generation_id) == []
    finally:
        conn.close()


def test_continuation_filter_drops_exact_parent_reel_id() -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["same-reel"], ["same-reel"]),
            (["same-reel"], ["same-reel"]),
        ],
    )
    source_generation_id = next(iter(rows))
    parent = rows[source_generation_id][0]
    child = rows[child_generation_id][0]
    _persist_released_feed_reel(
        conn,
        generation_id=source_generation_id,
        reel=parent,
    )
    try:
        assert main._filter_continuation_release_temporal_overlaps(
            conn,
            source_generation_id=source_generation_id,
            generation_id=child_generation_id,
            reels=[child],
        ) == []
    finally:
        conn.close()


@pytest.mark.parametrize("chain_length", [2, 40])
def test_continuation_history_queries_stay_bounded_as_chain_grows(
    chain_length: int,
) -> None:
    conn = _conn()
    source_generation_id, rows, jobs = _released_generation_chain(
        conn,
        [
            ([f"history-{index}"], [f"history-{index}"])
            for index in range(chain_length)
        ],
    )
    for generation_id, generation_reels in rows.items():
        for reel in generation_reels:
            _persist_released_feed_reel(
                conn,
                generation_id=generation_id,
                reel=reel,
            )
    current_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key=f"bounded-current-{chain_length}",
        generation_mode="slow",
        retrieval_profile="unified",
        source_generation_id=source_generation_id,
    )
    current = _released_feed_reel("current-non-overlap", chain_length + 10)
    _persist_released_feed_reel(
        conn,
        generation_id=current_generation_id,
        reel=current,
    )
    try:
        filter_statements: list[str] = []
        conn.set_trace_callback(filter_statements.append)
        filtered = main._filter_continuation_release_temporal_overlaps(
            conn,
            source_generation_id=source_generation_id,
            generation_id=current_generation_id,
            reels=[current],
        )
        conn.set_trace_callback(None)
        filter_reads = [
            statement
            for statement in filter_statements
            if statement.lstrip().upper().startswith(("SELECT", "WITH"))
        ]

        delivered_statements: list[str] = []
        conn.set_trace_callback(delivered_statements.append)
        delivered = main._continuation_delivered_reel_ids(
            conn,
            str(jobs[-1]["id"]),
        )
        conn.set_trace_callback(None)
        delivered_reads = [
            statement
            for statement in delivered_statements
            if statement.lstrip().upper().startswith(("SELECT", "WITH"))
        ]

        assert filtered == [current]
        assert len(filter_reads) <= 3
        assert delivered == sorted(
            f"history-{index}" for index in range(chain_length)
        )
        assert len(delivered_reads) <= 3
    finally:
        conn.set_trace_callback(None)
        conn.close()


def test_released_feed_keeps_non_overlapping_same_source_child(monkeypatch) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["prerequisite"], ["prerequisite"]),
            (["dependent"], ["dependent"]),
        ],
    )
    source_generation_id = next(iter(rows))
    prerequisite = rows[source_generation_id][0]
    prerequisite.update(video_id="shared-video", t_start=10.0, t_end=30.0)
    dependent = rows[child_generation_id][0]
    dependent.update(video_id="shared-video", t_start=30.0, t_end=55.0)
    _persist_released_feed_reel(
        conn,
        generation_id=source_generation_id,
        reel=prerequisite,
    )
    _persist_released_feed_reel(
        conn,
        generation_id=child_generation_id,
        reel=dependent,
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=source_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["prerequisite"]},
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=child_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["dependent"]},
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(conn, generation_id=child_generation_id)

        assert [reel["reel_id"] for reel in ranked] == [
            "prerequisite",
            "dependent",
        ]
    finally:
        conn.close()


def test_released_feed_keeps_overlapping_declared_lesson_chain(monkeypatch) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["prerequisite"], ["prerequisite"]),
            (["dependent"], ["dependent"]),
        ],
    )
    source_generation_id = next(iter(rows))
    prerequisite = rows[source_generation_id][0]
    prerequisite.update(
        video_id="shared-video",
        t_start=10.0,
        t_end=30.0,
        search_context={
            "selection_contract_version": "quality_silence_v39",
            "chain_id": "worked-example",
            "selection_candidate_id": "prerequisite-candidate",
        },
    )
    dependent = rows[child_generation_id][0]
    dependent.update(
        video_id="shared-video",
        t_start=10.0,
        t_end=50.0,
        search_context={
            "selection_contract_version": "quality_silence_v39",
            "chain_id": "worked-example",
            "selection_candidate_id": "dependent-candidate",
            "prerequisite_ids": ["prerequisite-candidate"],
        },
    )
    _persist_released_feed_reel(
        conn,
        generation_id=source_generation_id,
        reel=prerequisite,
    )
    _persist_released_feed_reel(
        conn,
        generation_id=child_generation_id,
        reel=dependent,
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(conn, generation_id=child_generation_id)

        assert [reel["reel_id"] for reel in ranked] == [
            "prerequisite",
            "dependent",
        ]
    finally:
        conn.close()


def test_released_feed_uses_release_as_allow_list_not_frozen_display_order(
    monkeypatch,
) -> None:
    conn = _conn()
    generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [(["released-first", "released-second"], [
            "released-second",
            "raw-private",
            "released-first",
        ])],
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(conn, generation_id=generation_id)

        assert [reel["reel_id"] for reel in ranked] == [
            "released-second",
            "released-first",
        ]
    finally:
        conn.close()


def test_released_feed_keeps_organizer_order_after_seen_reel_is_filtered(
    monkeypatch,
) -> None:
    conn = _conn()
    generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [(["first", "second", "third"], ["third", "second", "first"])],
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=generation_id,
        metadata={
            "version": 2,
            "ordered_reel_ids": ["first", "second", "third"],
        },
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    monkeypatch.setattr(
        main,
        "_learner_seen_reel_ids",
        lambda *_args, **_kwargs: {"first"},
    )
    try:
        ranked = _rank_released_feed(conn, generation_id=generation_id)

        assert [reel["reel_id"] for reel in ranked] == ["second", "third"]
    finally:
        conn.close()


def test_released_feed_keeps_reused_sibling_across_organizer_batches(
    monkeypatch,
) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["a", "b"], ["b", "c", "a"]),
            (["c", "d"], ["d"]),
        ],
    )
    source_generation_id = str(
        db.fetch_one(
            conn,
            "SELECT source_generation_id FROM reel_generations WHERE id = ?",
            (child_generation_id,),
        )["source_generation_id"]
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=source_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["a", "b"]},
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=child_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["c", "d"]},
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    monkeypatch.setattr(
        main,
        "_learner_seen_reel_ids",
        lambda *_args, **_kwargs: {"a"},
    )
    try:
        ranked = _rank_released_feed(
            conn,
            generation_id=child_generation_id,
        )

        assert [reel["reel_id"] for reel in ranked] == ["b", "c", "d"]
    finally:
        conn.close()


def test_released_feed_does_not_freeze_legacy_source_batch_to_release_order(
    monkeypatch,
) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["a", "b"], ["b", "a"]),
            (["c", "d"], ["d", "c"]),
        ],
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=child_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["c", "d"]},
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(
            conn,
            generation_id=child_generation_id,
        )

        assert [reel["reel_id"] for reel in ranked] == ["b", "a", "d", "c"]
    finally:
        conn.close()


def test_released_feed_rejects_unknown_organizer_metadata_before_projection(
    monkeypatch,
) -> None:
    conn = _conn()
    generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [(["a", "b"], ["b", "a"])],
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["unknown"]},
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(conn, generation_id=generation_id)

        assert [reel["reel_id"] for reel in ranked] == ["b", "a"]
    finally:
        conn.close()


def test_empty_child_release_does_not_disable_source_organizer_order(
    monkeypatch,
) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["a", "b"], ["b", "a"]),
            ([], ["raw-private"]),
        ],
    )
    source_generation_id = str(
        db.fetch_one(
            conn,
            "SELECT source_generation_id FROM reel_generations WHERE id = ?",
            (child_generation_id,),
        )["source_generation_id"]
    )
    main._persist_generation_lesson_order(
        conn,
        generation_id=source_generation_id,
        metadata={"version": 2, "ordered_reel_ids": ["a", "b"]},
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(
            conn,
            generation_id=child_generation_id,
        )

        assert [reel["reel_id"] for reel in ranked] == ["a", "b"]
    finally:
        conn.close()


def test_released_feed_treats_authoritative_empty_as_empty(monkeypatch) -> None:
    conn = _conn()
    generation_id, rows, _jobs = _released_generation_chain(
        conn, [([], ["raw-private"])]
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        assert _rank_released_feed(conn, generation_id=generation_id) == []
    finally:
        conn.close()


def test_released_feed_omits_missing_rows_without_raw_substitution(monkeypatch) -> None:
    conn = _conn()
    generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (
                ["released-present", "released-missing"],
                ["raw-private", "released-present"],
            )
        ],
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(conn, generation_id=generation_id)

        assert [reel["reel_id"] for reel in ranked] == ["released-present"]
    finally:
        conn.close()


def test_released_feed_stably_deduplicates_and_preserves_exclusion_order(
    monkeypatch,
) -> None:
    conn = _conn()
    child_generation_id, rows, _jobs = _released_generation_chain(
        conn,
        [
            (["r1", "r2"], ["r1", "r2"]),
            (["r2", "r3", "r4"], ["r2", "r3", "r4"]),
        ],
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    try:
        ranked = _rank_released_feed(
            conn,
            generation_id=child_generation_id,
            exclude_reel_ids=["r2"],
            exclude_video_ids=["video-r3"],
        )

        assert [reel["reel_id"] for reel in ranked] == ["r1", "r4"]
    finally:
        conn.close()


def test_terminal_status_and_replay_use_current_ranking_within_released_batch(
    monkeypatch,
) -> None:
    conn = _conn()
    _generation_id, rows, jobs = _released_generation_chain(
        conn,
        [
            (["r1", "r2"], ["r1", "r2"]),
            (["r4", "r3"], ["r3", "raw-private", "r4"]),
        ],
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    terminal = generation_jobs.get_job(conn, str(jobs[-1]["id"]))
    assert terminal is not None
    try:
        status = main._generation_job_status_payload(conn, terminal)
        replay = main._sanitize_generation_replay_events(
            conn,
            terminal,
            generation_jobs.replay_events(conn, job_id=str(terminal["id"])),
        )
        running_replay = main._sanitize_generation_replay_events(
            conn,
            {**terminal, "status": "running"},
            generation_jobs.replay_events(conn, job_id=str(terminal["id"])),
        )
        cancelled_replay = main._sanitize_generation_replay_events(
            conn,
            {**terminal, "status": "cancelled"},
            generation_jobs.replay_events(conn, job_id=str(terminal["id"])),
        )

        assert [reel["reel_id"] for reel in status["reels"]] == ["r3", "r4"]
        final = next(event for event in replay if event["type"] == "final")
        assert [reel["reel_id"] for reel in final["payload"]["reels"]] == [
            "r3", "r4",
        ]
        running_final = next(event for event in running_replay if event["type"] == "final")
        assert [reel["reel_id"] for reel in running_final["payload"]["reels"]] == [
            "r3", "r4",
        ]
        cancelled_final = next(event for event in cancelled_replay if event["type"] == "final")
        assert cancelled_final["payload"]["reels"] == []
        assert cancelled_final["payload"]["generation_id"] is None
    finally:
        conn.close()


def test_terminal_authoritative_empty_does_not_expose_raw_inventory(
    monkeypatch,
) -> None:
    conn = _conn()
    _generation_id, rows, jobs = _released_generation_chain(
        conn, [([], ["raw-private"])]
    )
    _patch_released_ranked_feed(monkeypatch, rows)
    terminal = generation_jobs.get_job(conn, str(jobs[-1]["id"]))
    assert terminal is not None
    try:
        status = main._generation_job_status_payload(conn, terminal)
        replay = main._sanitize_generation_replay_events(
            conn,
            terminal,
            generation_jobs.replay_events(conn, job_id=str(terminal["id"])),
        )

        assert status["reels"] == []
        final = next(event for event in replay if event["type"] == "final")
        assert final["payload"]["reels"] == []
    finally:
        conn.close()


def test_raw_generation_ranking_does_not_require_a_terminal_release(
    monkeypatch,
) -> None:
    conn = _conn()
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="running-raw-generation",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    monkeypatch.setattr(
        main,
        "_authoritative_release_reel_ids",
        lambda *_args, **_kwargs: pytest.fail(
            "raw worker ranking must not read terminal releases"
        ),
    )
    monkeypatch.setattr(
        main.reel_service,
        "ranked_feed",
        lambda *_args, **_kwargs: [_released_feed_reel("raw-candidate", 0)],
    )
    monkeypatch.setattr(
        main.reel_service,
        "learner_progress",
        lambda *_args, **_kwargs: {"selected_level": "beginner"},
    )
    try:
        ranked = _rank_released_feed(
            conn,
            generation_id=generation_id,
            released_only=False,
        )

        assert [reel["reel_id"] for reel in ranked] == ["raw-candidate"]
    finally:
        conn.close()


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
            "_selection_contract_version": "quality_silence_v39",
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
        assert reels[0]["selection_contract_version"] == "quality_silence_v39"
        assert reels[0]["relevance_score"] == 0.93
        assert reels[0]["topic_relevance"] == 0.93
        assert not any(key.startswith("_selection_") for key in reels[0])
    finally:
        conn.close()


def test_generation_job_reels_reuses_unseen_source_inventory_without_redelivery(
    monkeypatch,
) -> None:
    conn = _conn()
    observed: dict[str, object] = {}

    def ranked(*_args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(main, "_ranked_request_reels", ranked)
    monkeypatch.setattr(
        main,
        "get_generation_job",
        lambda _conn, job_id: {
            "id": job_id,
            "result_generation_id": "previous-generation",
            "request_params_json": "{}",
        } if job_id == "previous-job" else None,
    )
    monkeypatch.setattr(
        main,
        "_generation_chain_rows_snapshot",
        lambda *_args, **_kwargs: {
            "previous-generation": {
                "id": "previous-generation",
                "source_generation_id": None,
                "lesson_order_json": None,
            },
        },
    )
    monkeypatch.setattr(
        main,
        "_authoritative_generation_releases_snapshot",
        lambda *_args, **_kwargs: {
            "previous-generation": ["already-delivered"],
        },
    )
    try:
        reels = main._generation_job_reels(
            conn,
            {
                "id": "continued-job",
                "result_generation_id": "continued-generation",
                "material_id": "m1",
                "concept_id": "c1",
                "learner_id": "learner-1",
                "request_params_json": json.dumps({
                    "generation_mode": "slow",
                    "num_reels": 4,
                    "continuation_token": "previous-job",
                }),
            },
        )

        assert reels == []
        assert observed["generation_id"] == "continued-generation"
        assert observed["include_source_chain"] is True
        assert observed["exclude_reel_ids"] == ["already-delivered"]
    finally:
        conn.close()


def test_fast_final_uses_nearest_difficulty_without_hiding_valid_inventory() -> None:
    conn = _conn()
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="mixed-difficulty-fast",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    created_at = "2026-07-15T00:00:00+00:00"
    difficulties = [0.15, 0.25, *([0.50] * 6), *([0.85] * 5)]
    try:
        for index, difficulty in enumerate(difficulties):
            reel_id = f"mixed-reel-{index}"
            _insert_generation_reel(
                conn,
                generation_id=generation_id,
                reel_id=reel_id,
                video_id=f"mixed-video-{index}",
                created_at=created_at,
            )
            context = json.loads(conn.execute(
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel_id,),
            ).fetchone()[0])
            context.update({
                "source_rank": index,
                "boundary_confidence": 0.9,
                "surface_eligible": index < 2,
                "surface_reason": "" if index < 2 else "level_mismatch",
                "deferred_level": index >= 2,
            })
            conn.execute(
                "UPDATE reels SET difficulty = ?, search_context_json = ? WHERE id = ?",
                (difficulty, json.dumps(context), reel_id),
            )

        reels = main._generation_job_reels(
            conn,
            {
                "result_generation_id": generation_id,
                "material_id": "m1",
                "concept_id": "c1",
                "learner_id": "learner-1",
                "request_params_json": json.dumps({
                    "generation_mode": "fast",
                    "num_reels": 8,
                    "knowledge_level": "beginner",
                }),
            },
        )

        assert [reel["reel_id"] for reel in reels] == [
            f"mixed-reel-{index}" for index in range(8)
        ]
        assert main._count_generation_reels(conn, generation_id) == 13
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 13
        assert main._count_material_ready_reels(conn, "m1") == 13
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


def test_gemini_authority_reuse_bypasses_semantic_and_boundary_gates() -> None:
    conn = _conn()
    generation_id = "gemini-authority-generation"
    created_at = "2026-07-10T00:00:00+00:00"
    conn.execute(
        "INSERT INTO reel_generations "
        "(id, material_id, concept_id, request_key, generation_mode, "
        "retrieval_profile, status, reel_count, created_at) "
        "VALUES (?, 'm1', 'c1', 'gemini-authority', 'fast', 'deep', "
        "'completed', 1, ?)",
        (generation_id, created_at),
    )
    _insert_generation_reel(
        conn,
        generation_id=generation_id,
        reel_id="gemini-authority-reel",
        video_id="gemini-authority-video",
        created_at=created_at,
    )
    try:
        row = conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'gemini-authority-reel'"
        ).fetchone()
        context = json.loads(str(row["search_context_json"]))
        context.update({
            "selection_authority": "gemini",
            "informativeness": 0.1,
            "topic_relevance": 0.2,
            "educational_importance": 0.3,
            "directly_teaches_topic": False,
            "substantive": False,
            "factually_grounded": False,
            "self_contained": False,
            "is_standalone": False,
            "topic_evidence_quote": "",
            "surface_eligible": False,
            "surface_reason": "semantic_rejection",
            "deferred_level": True,
        })
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'gemini-authority-reel'",
            (json.dumps(context),),
        )

        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is True
        assert main._count_generation_reels(conn, generation_id) == 1
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 1
        assert main._count_material_ready_reels(conn, "m1") == 1
        assert main._usable_boundary_reel_ids(
            conn, ["gemini-authority-reel"]
        ) == {"gemini-authority-reel"}

        context.update({
            "boundary_status": "failed",
            "boundary_diagnostics": {"acoustic_verified": False},
        })
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'gemini-authority-reel'",
            (json.dumps(context),),
        )
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is True
        assert main._count_generation_reels(conn, generation_id) == 1
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 1
        assert main._count_material_ready_reels(conn, "m1") == 1
        assert main._usable_boundary_reel_ids(
            conn, ["gemini-authority-reel"]
        ) == {"gemini-authority-reel"}

        conn.execute(
            "UPDATE reels SET t_start = -10, t_end = -1 "
            "WHERE id = 'gemini-authority-reel'"
        )
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is False
        assert main._count_generation_reels(conn, generation_id) == 0
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 0
        assert main._count_material_ready_reels(conn, "m1") == 0
        assert main._usable_boundary_reel_ids(
            conn, ["gemini-authority-reel"]
        ) == set()

        conn.execute(
            "UPDATE reels SET t_start = 10, t_end = 10 "
            "WHERE id = 'gemini-authority-reel'"
        )
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is False
        assert main._count_generation_reels(conn, generation_id) == 0
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 0
        assert main._count_material_ready_reels(conn, "m1") == 0
        assert main._usable_boundary_reel_ids(
            conn, ["gemini-authority-reel"]
        ) == set()
    finally:
        conn.close()


def test_transcript_aligned_inventory_treats_legacy_level_mismatch_as_soft() -> None:
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
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 1
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
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 0
        assert main._usable_boundary_reel_ids(
            conn, ["transcript-boundary-reel"]
        ) == set()

        context["surface_reason"] = "level_mismatch"
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (json.dumps(context), "transcript-boundary-reel"),
        )
        assert main._count_generation_reels(conn, generation_id) == 1
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 1
        assert main._usable_boundary_reel_ids(
            conn, ["transcript-boundary-reel"]
        ) == {"transcript-boundary-reel"}

        context["selection_contract_version"] = "quality_silence_v15"
        conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = ?",
            (json.dumps(context), "transcript-boundary-reel"),
        )
        assert main._count_generation_reels(conn, generation_id) == 0
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 0
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


def test_generation_worker_cancels_inventory_acquired_under_stale_adaptation(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="stale-adaptation-release",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 1,
            "knowledge_level": "beginner",
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
        },
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="stale-adaptation-worker",
        now=now,
    )
    assert leased
    concept_signals: dict[str, dict[str, float]] = {}
    ordering_calls = 0

    def generate_then_receive_feedback(worker_conn, **kwargs) -> None:
        nonlocal concept_signals
        _insert_generation_reel(
            worker_conn,
            generation_id=str(kwargs["generation_id"]),
            reel_id="pre-feedback-reel",
            video_id="pre-feedback-video",
            created_at=now.isoformat(),
        )
        concept_signals = {
            "c1": {"helpful": 0.0, "confusing": 1.0, "adjustment": -0.35}
        }

    def order_batch(reels, **_kwargs):
        nonlocal ordering_calls
        ordering_calls += 1
        return mock.Mock(
            reels=reels,
            ordered_reel_ids=[reel["reel_id"] for reel in reels],
            assessment_checkpoint_reel_ids=[],
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_then_receive_feedback)
    monkeypatch.setattr(
        main,
        "_learner_concept_signals",
        lambda *_args, **_kwargs: concept_signals,
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        cancelled = generation_jobs.get_job(conn, job["id"])
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        assert cancelled["terminal_error_code"] == "cancelled"
        assert ordering_calls == 0
        assert not any(
            event["type"] == "final"
            for event in generation_jobs.replay_events(conn, job_id=job["id"])
        )
        assert main._generation_job_status_payload(conn, cancelled)["reels"] == []
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_heads").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize("stale_dimension", ["concept", "knowledge_level"])
def test_generation_worker_rejects_stale_lease_before_provider_work(
    monkeypatch,
    stale_dimension: str,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="stale-adaptation-preflight",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 1,
            "knowledge_level": "beginner",
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
        },
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="stale-preflight-worker",
        now=now,
    )
    assert leased
    provider = mock.Mock()
    monkeypatch.setattr(main.reel_service, "generate_reels", provider)
    if stale_dimension == "concept":
        monkeypatch.setattr(
            main,
            "_learner_concept_signals",
            lambda *_args, **_kwargs: {
                "c1": {"helpful": 0.0, "confusing": 1.0, "adjustment": -0.35}
            },
        )
    else:
        monkeypatch.setattr(
            main.reel_service,
            "learner_progress",
            lambda *_args, **_kwargs: {
                "selected_level": "advanced",
                "global_adjustment": 0.0,
            },
        )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        cancelled = generation_jobs.get_job(conn, job["id"])
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        provider.assert_not_called()
        assert conn.execute("SELECT COUNT(*) FROM reel_generations").fetchone()[0] == 0
    finally:
        conn.close()


def test_generation_worker_rechecks_adaptation_at_atomic_release(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="stale-adaptation-final-release",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 1,
            "knowledge_level": "beginner",
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
        },
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="stale-release-worker",
        now=now,
    )
    assert leased
    concept_signals: dict[str, dict[str, float]] = {}
    original_lock = main._lock_learner_adaptation

    def generate_stage(worker_conn, **kwargs) -> None:
        _insert_generation_reel(
            worker_conn,
            generation_id=str(kwargs["generation_id"]),
            reel_id="pre-release-feedback-reel",
            video_id="pre-release-feedback-video",
            created_at=now.isoformat(),
        )

    def order_batch(reels, **_kwargs):
        return mock.Mock(
            reels=reels,
            ordered_reel_ids=[reel["reel_id"] for reel in reels],
            assessment_checkpoint_reel_ids=[],
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    def lock_after_feedback(worker_conn, **kwargs) -> None:
        nonlocal concept_signals
        original_lock(worker_conn, **kwargs)
        concept_signals = {
            "c1": {"helpful": 1.0, "confusing": 0.0, "adjustment": 0.2}
        }

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    monkeypatch.setattr(main, "_lock_learner_adaptation", lock_after_feedback)
    monkeypatch.setattr(
        main,
        "_learner_concept_signals",
        lambda *_args, **_kwargs: concept_signals,
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        cancelled = generation_jobs.get_job(conn, job["id"])
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        assert not any(
            event["type"] == "final"
            for event in generation_jobs.replay_events(conn, job_id=job["id"])
        )
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_heads").fetchone()[0] == 0
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
                        "selection_contract_version": "quality_silence_v39",
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


def test_off_level_generation_releases_valid_inventory_without_second_search(
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
            "boundary_confidence": 0.9,
            "selection_candidate_id": "deferred-only-video::deferred-unit",
        })
        worker_conn.execute(
            "UPDATE reels SET search_context_json = ? WHERE id = 'deferred-only-reel'",
            (json.dumps(context),),
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_deferred_reel)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        terminal_job = generation_jobs.get_job(conn, job["id"])
        assert terminal_job is not None
        generation_id = str(terminal_job["result_generation_id"] or "")
        assert terminal_job["status"] == "completed"
        assert generation_id
        assert terminal_job["terminal_error_code"] is None
        assert main._count_generation_reels(conn, generation_id) == 1
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 1
        stored_context = json.loads(str(conn.execute(
            "SELECT search_context_json FROM reels WHERE id = 'deferred-only-reel'"
        ).fetchone()["search_context_json"]))
        assert stored_context["surface_eligible"] is False
        assert stored_context["surface_reason"] == "level_mismatch"
        assert stored_context["deferred_level"] is True

        generation = conn.execute(
            "SELECT status, reel_count, error_text FROM reel_generations WHERE id = ?",
            (generation_id,),
        ).fetchone()
        assert tuple(generation) == ("active", 1, None)
        head = conn.execute(
            "SELECT active_generation_id FROM reel_generation_heads "
            "WHERE material_id = 'm1' AND request_key = 'deferred-only-request'"
        ).fetchone()
        assert head["active_generation_id"] == generation_id

        events = generation_jobs.replay_events(conn, job_id=job["id"])
        final_event = next(event for event in events if event["type"] == "final")
        terminal_event = next(event for event in events if event["type"] == "terminal")
        assert [
            reel["reel_id"] for reel in final_event["payload"]["reels"]
        ] == ["deferred-only-reel"]
        assert final_event["payload"]["generation_id"] == generation_id
        assert final_event["payload"]["authoritative"] is True
        assert terminal_event["payload"]["status"] == "completed"
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


def test_generation_worker_releases_nearest_level_clip_when_it_is_only_valid_clip(
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
            "boundary_confidence": 0.9,
            "selection_candidate_id": (
                "usable-intermediate-video::nearest-level-unit"
            ),
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
    try:
        main._run_leased_generation_job(leased, threading.Event())

        terminal_job = generation_jobs.get_job(conn, job["id"])
        assert terminal_job is not None
        assert terminal_job["status"] == "partial"
        assert terminal_job["terminal_error_code"] is None
        generation_id = str(terminal_job["result_generation_id"] or "")
        assert generation_id
        assert main._count_generation_reels(conn, generation_id) == 1
        assert main._count_generation_surfaceable_reels(conn, generation_id) == 1
        assert main._verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id="m1",
        ) is True

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


def test_deferred_only_slow_reservoir_queues_current_batch_after_level_change(
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
    monkeypatch.setattr(
        main,
        "_require_verified_provider_account",
        lambda *_args, **_kwargs: {"id": "account-1"},
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
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
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
        generation_response = asyncio.run(main.generate_reels(
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
        assert feed_response["generation_job_id"]
        assert isinstance(generation_response, JSONResponse)
        assert generation_response.status_code == 202
        current_job_id = json.loads(generation_response.body)["job_id"]
        assert current_job_id == feed_response["generation_job_id"]
        current_job = generation_jobs.get_job(conn, current_job_id)
        assert current_job is not None
        assert current_job["source_generation_id"] == source_generation_id
        assert main._job_request_params(current_job)["fresh_source_budget"] is True
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == initial_jobs + 1
        assert conn.execute(
            "SELECT COUNT(*) FROM generation_provider_usage"
        ).fetchone()[0] == initial_provider_rows
        provider.assert_not_called()
        assert wake.call_args_list == [mock.call(), mock.call()]
    finally:
        conn.close()


def test_cross_request_legacy_level_reservoir_does_not_force_fresh_top_up(
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
    for index in range(5):
        reel_id = f"legacy-level-reel-{index}"
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=reel_id,
            video_id=f"legacy-level-video-{index}",
            created_at=(now + timedelta(seconds=index)).isoformat(),
        )
        row = conn.execute(
            "SELECT search_context_json FROM reels WHERE id = ?",
            (reel_id,),
        ).fetchone()
        context = json.loads(str(row[0]))
        context.update({
            "surface_eligible": False,
            "surface_reason": "level_mismatch",
            "deferred_level": True,
            "boundary_confidence": 0.9,
        })
        conn.execute(
            "UPDATE reels SET difficulty = 0.85, search_context_json = ? "
            "WHERE id = ?",
            (json.dumps(context), reel_id),
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
    organizer_candidates: list[str] = []

    def order_batch(reels, **_kwargs):
        organizer_candidates.extend(str(reel["reel_id"]) for reel in reels)
        selected = list(reels)
        return mock.Mock(
            reels=selected,
            ordered_reel_ids=[str(reel["reel_id"]) for reel in selected],
            assessment_checkpoint_reel_ids=[],
            model_used="gemini-test",
            degraded=False,
            fallback_reason=None,
            provider_called=True,
        )

    generate_stage = mock.Mock(
        side_effect=AssertionError("soft level inventory must prevent retrieval")
    )
    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
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
    try:
        assert main._current_level_reusable_generation_reel_count(
            conn,
            generation_id=source_generation_id,
            material_id="m1",
            concept_id="c1",
            learner_id="learner-1",
            request_params={"generation_mode": "slow", "num_reels": 5},
            requested=5,
        ) == 5

        main._run_leased_generation_job(leased, threading.Event())

        generate_stage.assert_not_called()
        assert set(organizer_candidates) == {
            f"legacy-level-reel-{index}" for index in range(5)
        }
        assert generation_jobs.get_job(conn, job["id"])["status"] == "completed"
    finally:
        conn.close()


def test_partial_cross_request_startup_uses_fresh_bounded_source_budget(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="partial-startup-source",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="partial-startup-current",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 9,
            "fresh_source_budget": True,
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-partial-startup",
        now=now,
    )
    assert leased
    calls: list[dict] = []

    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 3,
    )
    monkeypatch.setattr(
        main,
        "_generation_chain_analyzed_source_budget",
        lambda *_args, **_kwargs: main.GENERATION_SOURCE_BUDGETS["fast"],
    )
    monkeypatch.setattr(
        main.reel_service,
        "generate_reels",
        lambda _worker_conn, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": f"strict-cached-reel-{index}"}
            for index in range(3)
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert len(calls) == 1
        assert calls[0]["num_reels"] == 9
        assert calls[0]["max_generation_videos"] == main.GENERATION_SOURCE_BUDGETS["fast"]
        assert calls[0]["max_new_reels"] == 6
        assert generation_jobs.get_job(conn, job["id"])["status"] == "partial"
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("reusable_count", "expected_new_reels", "expected_status"),
    [(3, 0, "completed"), (1, 0, "partial"), (0, 3, "completed")],
)
def test_continuation_returns_any_unseen_cache_before_a_fresh_source_budget(
    monkeypatch,
    reusable_count: int,
    expected_new_reels: int,
    expected_status: str,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="continuation-source",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="continuation-source",
        generation_id=source_generation_id,
        completed_at=now.isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "slow", "num_reels": 3},
    )
    conn.execute(
        "UPDATE reel_generation_jobs SET usage_json = ? WHERE id = ?",
        (json.dumps({"counters": {"analyzed_sources": 3}}), prior_job["id"]),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="continuation-batch",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "slow",
            "num_reels": 3,
            "continuation_token": prior_job["id"],
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-continuation",
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
        lambda *_args, **_kwargs: reusable_count,
    )
    monkeypatch.setattr(
        main,
        "_count_generation_surfaceable_reels",
        lambda _conn, generation_id: (
            0 if generation_id == source_generation_id else generated_count
        ),
    )
    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {"reel_id": f"continued-reel-{index}"}
            for index in range(reusable_count + generated_count)
        ],
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        assert [call["max_new_reels"] for call in calls] == (
            [expected_new_reels] if expected_new_reels else []
        )
        if calls:
            assert calls[0]["max_generation_videos"] == 3
            assert source_generation_id in calls[0]["exclude_generation_ids"]
        assert generation_jobs.get_job(conn, job["id"])["status"] == expected_status
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("provider_cursor_open", "expected_status"),
    [(False, "exhausted"), (True, "partial")],
)
def test_empty_continuation_status_reflects_provider_exhaustion(
    monkeypatch, provider_cursor_open: bool, expected_status: str,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="continuation-inventory-source",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=source_generation_id,
        reel_id="prior-batch-reel",
        video_id="prior-batch-video",
        created_at=now.isoformat(),
    )
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="continuation-inventory-source",
        generation_id=source_generation_id,
        completed_at=now.isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "slow", "num_reels": 1},
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="empty-continuation",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "slow",
            "num_reels": 1,
            "continuation_token": prior_job["id"],
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="worker-empty-continuation",
        now=now,
    )
    assert leased

    def generate_reels(*_args, **kwargs) -> None:
        if provider_cursor_open:
            kwargs["generation_context"].increment_counter("provider_cursor_open")

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_reels)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed = generation_jobs.get_job(conn, job["id"])
        assert completed and completed["status"] == expected_status
        events = generation_jobs.replay_events(conn, job_id=job["id"])
        terminal = events[-1]
        assert terminal["payload"]["status"] == expected_status
        final = next(event for event in events if event["type"] == "final")
        assert bool(final["payload"]["generation_id"]) is provider_cursor_open
        generation = main._fetch_generation_row(
            conn, str(completed["result_generation_id"])
        )
        assert generation and generation["status"] == (
            "partial" if provider_cursor_open else "failed"
        )
    finally:
        conn.close()


def test_continuations_carry_consumed_sources_and_stop_after_cursor_exhaustion(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="consumed-source-root",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="consumed-source-root",
        generation_id=source_generation_id,
        completed_at=now.isoformat(),
        status="partial",
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "fast", "num_reels": 3},
    )
    conn.execute(
        "UPDATE reel_generation_jobs SET usage_json = ? WHERE id = ?",
        (
            json.dumps({
                "consumed_video_ids": [
                    "AAAAAAAAAAA",
                    "yt:BBBBBBBBBBB",
                    "not-a-video-id",
                ],
            }),
            prior_job["id"],
        ),
    )

    first_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="consumed-source-first-continuation",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 3,
            "continuation_token": prior_job["id"],
        },
        source_generation_id=source_generation_id,
        now=now,
    )
    first_lease = generation_jobs.lease_job(
        conn,
        job_id=first_job["id"],
        lease_owner="worker-consumed-first",
        now=now,
    )
    assert first_lease
    calls: list[dict] = []

    def generate_stage(_worker_conn, **kwargs) -> None:
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["retrieved_video_ids"].update({
                "CCCCCCCCCCC",
                "yt:DDDDDDDDDDD",
            })
            kwargs["attempted_video_ids"].update({
                "CCCCCCCCCCC",
                "yt:DDDDDDDDDDD",
                "EEEEEEEEEEE",
            })
            kwargs["analyzed_video_ids"].add("CCCCCCCCCCC")
            kwargs["capacity_deferred_video_ids"].add("EEEEEEEEEEE")
            kwargs["generation_context"].increment_counter(
                "provider_cursor_open"
            )
        elif len(calls) == 2:
            assert "DDDDDDDDDDD" not in kwargs["exclude_video_ids"]
            kwargs["retrieved_video_ids"].add("DDDDDDDDDDD")
            kwargs["attempted_video_ids"].add("DDDDDDDDDDD")
            kwargs["generation_context"].increment_counter(
                "provider_cursor_open"
            )

    monkeypatch.setattr(
        main,
        "_current_level_reusable_generation_reel_count",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        main,
        "_count_generation_surfaceable_reels",
        lambda *_args: 0,
    )
    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(main, "_generation_job_reels", lambda *_args, **_kwargs: [])
    try:
        main._run_leased_generation_job(first_lease, threading.Event())

        first_completed = generation_jobs.get_job(conn, first_job["id"])
        assert first_completed and first_completed["status"] == "partial"
        assert calls[0]["consumed_video_ids"] == [
            "AAAAAAAAAAA",
            "BBBBBBBBBBB",
        ]
        first_usage = json.loads(str(first_completed["usage_json"] or "{}"))
        assert first_usage["consumed_video_ids"] == ["CCCCCCCCCCC"]
        assert first_usage["capacity_deferred_video_ids"] == ["EEEEEEEEEEE"]
        assert first_usage["failed_source_attempts"] == {"DDDDDDDDDDD": 1}
        first_generation_id = str(first_completed["result_generation_id"] or "")
        assert main._generation_chain_consumed_video_ids(
            conn,
            generation_id=first_generation_id,
        ) == {
            "AAAAAAAAAAA",
            "BBBBBBBBBBB",
            "CCCCCCCCCCC",
        }

        second_job, _ = generation_jobs.submit_or_get_active(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key="consumed-source-second-continuation",
            content_fingerprint="fingerprint",
            learner_id="learner-1",
            request_params={
                "generation_mode": "fast",
                "num_reels": 3,
                "continuation_token": first_job["id"],
            },
            source_generation_id=first_generation_id,
            now=now + timedelta(seconds=1),
        )
        second_lease = generation_jobs.lease_job(
            conn,
            job_id=second_job["id"],
            lease_owner="worker-consumed-second",
            now=now + timedelta(seconds=1),
        )
        assert second_lease
        main._run_leased_generation_job(second_lease, threading.Event())

        assert calls[1]["consumed_video_ids"] == [
            "AAAAAAAAAAA",
            "BBBBBBBBBBB",
            "CCCCCCCCCCC",
        ]
        second_completed = generation_jobs.get_job(conn, second_job["id"])
        assert second_completed and second_completed["status"] == "partial"
        second_usage = json.loads(str(second_completed["usage_json"] or "{}"))
        assert second_usage["failed_source_attempts"] == {"DDDDDDDDDDD": 1}

        second_generation_id = str(second_completed["result_generation_id"] or "")
        assert main._generation_chain_failed_source_attempts(
            conn,
            generation_id=second_generation_id,
        ) == {"DDDDDDDDDDD": 2}

        third_job, _ = generation_jobs.submit_or_get_active(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key="consumed-source-third-continuation",
            content_fingerprint="fingerprint",
            learner_id="learner-1",
            request_params={
                "generation_mode": "fast",
                "num_reels": 3,
                "continuation_token": second_job["id"],
            },
            source_generation_id=second_generation_id,
            now=now + timedelta(seconds=2),
        )
        third_lease = generation_jobs.lease_job(
            conn,
            job_id=third_job["id"],
            lease_owner="worker-consumed-third",
            now=now + timedelta(seconds=2),
        )
        assert third_lease
        main._run_leased_generation_job(third_lease, threading.Event())

        assert calls[2]["consumed_video_ids"] == [
            "AAAAAAAAAAA",
            "BBBBBBBBBBB",
            "CCCCCCCCCCC",
        ]
        assert calls[2]["exclude_video_ids"] == ["DDDDDDDDDDD"]
        third_completed = generation_jobs.get_job(conn, third_job["id"])
        assert third_completed and third_completed["status"] == "exhausted"
    finally:
        conn.close()


def test_fast_source_actual_completion_count_controls_slow_top_up(
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
    prior_job = _terminal_job_for_generation(
        conn,
        request_key="prior-fast-source-budget",
        generation_id=source_generation_id,
        completed_at=now.isoformat(),
        content_fingerprint="fingerprint",
        request_params={"generation_mode": "fast", "num_reels": 8},
    )
    conn.execute(
        "UPDATE reel_generation_jobs SET usage_json = ? WHERE id = ?",
        (
            json.dumps({"counters": {"analyzed_sources": 1}}),
            prior_job["id"],
        ),
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
        "_count_generation_surfaceable_reels",
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
        assert calls[0]["max_generation_videos"] == 2
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
            call["generation_context"].require_acoustic_boundaries
            for call in calls
        )
        assert calls[0]["exclude_video_ids"] == ["manual-exclusion"]
        assert ("retrieval", 0.05) in progress_updates
        assert ("ranking", 0.85) in progress_updates
        events = generation_jobs.replay_events(conn, job_id=job["id"])
        assert [event["type"] for event in events] == [
            "final",
            "terminal",
        ]
        assert [
            event["payload"]["reel"]["reel_id"]
            for event in events
            if event["type"] == "candidate"
        ] == []
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
    [("fast", 9, 2), ("slow", 9, 3)],
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
                "selection_contract_version": "quality_silence_v39",
            }
            kwargs["on_reel_created"](reel)
        generated_count += int(kwargs["max_new_reels"])

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_count_generation_surfaceable_reels",
        lambda *_args, **_kwargs: generated_count,
    )
    monkeypatch.setattr(
        main,
        "_generation_job_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": f"{mode}-reel-{index}",
                "selection_contract_version": "quality_silence_v39",
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
        assert calls[0]["generation_context"].require_acoustic_boundaries is True
        gemini_budget = calls[0]["generation_context"].budget.snapshot()["gemini"]
        assert gemini_budget["flash_selector_calls"] == expected_source_cap
        assert gemini_budget["flash_selector_limit"] == expected_source_cap
        assert gemini_budget["pro_calls"] == 0
        assert gemini_budget["pro_call_limit"] == 0
        events = generation_jobs.replay_events(conn, job_id=job["id"])
        assert [event["type"] for event in events].count("candidate") == 0
        assert [event["type"] for event in events][-2:] == ["final", "terminal"]
        final_event = next(event for event in events if event["type"] == "final")
        terminal_event = next(event for event in events if event["type"] == "terminal")
        assert len(final_event["payload"]["reels"]) == expected_reel_cap
        assert terminal_event["payload"]["status"] == "completed"
        completed_job = generation_jobs.get_job(conn, job["id"])
        assert completed_job is not None
        usage = json.loads(str(completed_job["usage_json"] or "{}"))
        assert usage["counters"]["analyzed_sources"] == expected_source_cap
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
        assert candidates == []
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
def test_single_stage_retryable_provider_failure_is_requeued(
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
        kwargs["retrieved_video_ids"].add("EEEEEEEEEEE")
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
        assert completed_job and completed_job["status"] == "queued"
        assert completed_job["attempt_count"] == 1
        assert [call["retrieval_profile"] for call in calls] == ["deep"]
        assert calls[0]["max_generation_videos"] == (2 if mode == "fast" else 3)
        terminal_error_code = conn.execute(
            "SELECT terminal_error_code FROM reel_generation_jobs WHERE id = ?",
            (job["id"],),
        ).fetchone()[0]
        assert terminal_error_code is None
        failed_usage = json.loads(str(completed_job["usage_json"] or "{}"))
        assert failed_usage["consumed_video_ids"] == []
        assert failed_usage["retry_errors"][0]["code"] == "provider_transient"
        assert main._generation_chain_consumed_video_ids(
            conn,
            generation_id=str(completed_job["result_generation_id"] or ""),
        ) == set()
        assert generation_jobs.replay_events(conn, job_id=job["id"]) == []
    finally:
        conn.close()


def test_single_stage_nonretryable_provider_failure_remains_fatal(monkeypatch) -> None:
    provider_error = ProviderQuotaError(
        "Supadata quota is exhausted.",
        provider="supadata",
        operation="search",
    )
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


def test_rate_limited_provider_job_honors_retry_after_before_second_lease(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-rate-limit-retry-after",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="provider-rate-limit-worker",
        now=now,
    )
    assert leased

    def fail_stage(_worker_conn, **_kwargs) -> None:
        raise ProviderRateLimitError(
            "Provider asked the client to retry later.",
            provider="gemini",
            operation="segmentation",
            status_code=429,
            retry_after_sec=5.0,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", fail_stage)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        requeued = generation_jobs.get_job(conn, job["id"])
        assert requeued and requeued["status"] == "queued"
        retry_at = datetime.fromisoformat(requeued["retry_not_before_at"])
        checked_at = datetime.now(timezone.utc)
        assert 4.0 <= (retry_at - checked_at).total_seconds() <= 5.0
        assert generation_jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="provider-rate-limit-too-early",
            now=checked_at,
        ) is None
        second = generation_jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="provider-rate-limit-worker-2",
            now=retry_at,
        )
        assert second and second["attempt_count"] == 2
    finally:
        conn.close()


def test_ingestion_process_rate_limit_requeues_same_job_until_retry_after(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="ingestion-rate-limit-retry-after",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="ingestion-rate-limit-worker",
        now=now,
    )
    assert leased

    def fail_stage(_worker_conn, **_kwargs) -> None:
        raise main.IngestRateLimitedError(
            "The process-wide YouTube limiter is saturated.",
            retry_after_sec=5.0,
            detail="youtube concurrency limit",
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", fail_stage)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        requeued = generation_jobs.get_job(conn, job["id"])
        assert requeued and requeued["status"] == "queued"
        assert requeued["phase"] == "retrying"
        assert requeued["terminal_error_code"] is None
        retry_at = datetime.fromisoformat(requeued["retry_not_before_at"])
        checked_at = datetime.now(timezone.utc)
        assert 4.0 <= (retry_at - checked_at).total_seconds() <= 5.0
        usage = json.loads(str(requeued["usage_json"] or "{}"))
        assert usage["retry_errors"][-1] == {
            "code": "provider_rate_limited",
            "message": "The process-wide YouTube limiter is saturated.",
            "provider": "ingestion",
            "operation": "platform_rate_limit",
            "retryable": True,
            "status_code": 429,
            "retry_after_sec": 5.0,
            "detail": "youtube concurrency limit",
        }
        assert generation_jobs.replay_events(conn, job_id=job["id"]) == []
        assert generation_jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="ingestion-rate-limit-too-early",
            now=checked_at,
        ) is None
        second = generation_jobs.lease_job(
            conn,
            job_id=job["id"],
            lease_owner="ingestion-rate-limit-worker-2",
            now=retry_at,
        )
        assert second and second["attempt_count"] == 2
    finally:
        conn.close()


def test_provider_failure_yields_and_wakes_when_lease_expires_during_call(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="provider-failure-expired-lease",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 1},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="provider-expiring-lease-worker",
        now=now,
    )
    assert leased
    wake = mock.Mock()

    def expire_then_fail(worker_conn, **_kwargs) -> None:
        worker_conn.execute(
            "UPDATE reel_generation_jobs SET lease_expires_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), job["id"]),
        )
        raise ProviderTransientError(
            "Provider failed as the worker lease expired.",
            provider="gemini",
            operation="segmentation",
            status_code=503,
        )

    monkeypatch.setattr(main.reel_service, "generate_reels", expire_then_fail)
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        yielded = generation_jobs.get_job(conn, job["id"])
        assert yielded and yielded["status"] == "running"
        assert yielded["attempt_count"] == 1
        wake.assert_called_once_with()
        recovered = generation_jobs.lease_next_job(
            conn,
            lease_owner="provider-expired-lease-recovery-worker",
            now=datetime.now(timezone.utc),
        )
        assert recovered and recovered["id"] == job["id"]
        assert recovered["attempt_count"] == 2
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


def test_generation_job_endpoints_hide_other_learners_jobs(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="learner-one-private-job",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={},
    )
    monkeypatch.setattr(
        main,
        "_resolve_learner_identity",
        lambda *_args, **_kwargs: "learner-2",
    )
    try:
        with pytest.raises(main.HTTPException) as status_error:
            main.generation_status(object(), job["id"])
        assert status_error.value.status_code == 404

        with pytest.raises(main.HTTPException) as stream_error:
            asyncio.run(main.generation_stream(object(), job["id"]))
        assert stream_error.value.status_code == 404

        with pytest.raises(main.HTTPException) as cancel_error:
            main.cancel_generation_job(object(), job["id"])
        assert cancel_error.value.status_code == 404
        assert generation_jobs.get_job(conn, job["id"])["status"] == "queued"
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


def test_generate_does_not_coalesce_jobs_across_learners(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    current_learner = ["learner-1"]
    monkeypatch.setattr(
        main,
        "_resolve_learner_identity",
        lambda *_args, **_kwargs: current_learner[0],
    )
    payload = ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=3)
    try:
        first = asyncio.run(main.generate_reels(object(), payload))
        current_learner[0] = "learner-2"
        second = asyncio.run(main.generate_reels(object(), payload))

        first_job_id = json.loads(first.body)["job_id"]
        second_job_id = json.loads(second.body)["job_id"]
        assert first_job_id != second_job_id
        jobs_by_id = {
            row["id"]: row["learner_id"]
            for row in conn.execute(
                "SELECT id, learner_id FROM reel_generation_jobs ORDER BY created_at, id"
            ).fetchall()
        }
        assert jobs_by_id == {
            first_job_id: "learner-1",
            second_job_id: "learner-2",
        }
    finally:
        conn.close()


def test_generate_continuation_queues_one_new_batch_from_the_previous_job(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    root_payload = ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=3)
    try:
        root_response = asyncio.run(main.generate_reels(object(), root_payload))
        root_job_id = json.loads(root_response.body)["job_id"]
        root_job = generation_jobs.get_job(conn, root_job_id)
        assert root_job
        root_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(root_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'completed', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (root_generation_id, completed_at, completed_at, root_job_id),
        )

        continuation_payload = ReelsGenerateRequest(
            material_id="m1",
            concept_id="c1",
            num_reels=3,
            continuation_token=root_job_id,
        )
        first = asyncio.run(main.generate_reels(object(), continuation_payload))
        repeated = asyncio.run(main.generate_reels(object(), continuation_payload))

        first_body = json.loads(first.body)
        repeated_body = json.loads(repeated.body)
        assert first.status_code == repeated.status_code == 202
        assert first_body["job_id"] == repeated_body["job_id"]
        assert first_body["job_id"] != root_job_id
        continuation_job = generation_jobs.get_job(conn, first_body["job_id"])
        assert continuation_job
        assert continuation_job["source_generation_id"] == root_generation_id
        continuation_params = json.loads(continuation_job["request_params_json"])
        assert continuation_params["continuation_token"] == root_job_id
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_generate_implicit_retry_carries_failed_source_attempts_and_stops_at_bound(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    payload = ReelsGenerateRequest(
        material_id="m1",
        concept_id="c1",
        num_reels=3,
    )
    failed_video_id = "AAAAAAAAAAA"
    try:
        first = asyncio.run(main.generate_reels(object(), payload))
        first_job_id = json.loads(first.body)["job_id"]
        first_job = generation_jobs.get_job(conn, first_job_id)
        assert first_job
        first_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(first_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=first_job_id,
            status="failed",
            result_generation_id=first_generation_id,
            usage={
                "counters": {"provider_cursor_open": 0},
                "consumed_video_ids": [],
                "failed_source_attempts": {failed_video_id: 1},
            },
        )

        retry = asyncio.run(main.generate_reels(object(), payload))
        assert retry.status_code == 202
        retry_job_id = json.loads(retry.body)["job_id"]
        retry_job = generation_jobs.get_job(conn, retry_job_id)
        assert retry_job
        assert retry_job["source_generation_id"] == first_generation_id

        retry_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(retry_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
            source_generation_id=first_generation_id,
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=retry_job_id,
            status="failed",
            result_generation_id=retry_generation_id,
            usage={
                "counters": {"provider_cursor_open": 0},
                "consumed_video_ids": [],
                "failed_source_attempts": {failed_video_id: 1},
            },
        )

        with pytest.raises(main.HTTPException) as exc_info:
            asyncio.run(main.generate_reels(object(), payload))
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "source_retry_exhausted"
        assert main._generation_chain_failed_source_attempts(
            conn,
            generation_id=retry_generation_id,
        ) == {failed_video_id: 2}
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_global_provider_failure_allows_explicit_root_but_suppresses_feed_retry(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    monkeypatch.setattr(main, "_ranked_request_reels", lambda *_args, **_kwargs: [])
    direct_payload = ReelsGenerateRequest(
        material_id="m1",
        concept_id="c1",
        num_reels=3,
    )
    try:
        first_direct = asyncio.run(main.generate_reels(object(), direct_payload))
        direct_job_id = json.loads(first_direct.body)["job_id"]
        direct_job = generation_jobs.get_job(conn, direct_job_id)
        assert direct_job
        direct_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(direct_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=direct_job_id,
            status="failed",
            result_generation_id=direct_generation_id,
            error_code="provider_quota_exhausted",
            usage={
                "counters": {"provider_cursor_open": 1},
                "consumed_video_ids": [],
                "failed_source_attempts": {"AAAAAAAAAAA": 1},
            },
        )
        direct_terminal = generation_jobs.get_job(conn, direct_job_id)
        assert direct_terminal
        assert main._generation_job_has_retryable_source_work(
            conn,
            direct_terminal,
        ) is False

        cross_request_lookup = mock.Mock(return_value="must-not-link")
        monkeypatch.setattr(
            main,
            "_verified_cross_request_source_generation",
            cross_request_lookup,
        )
        direct_retry = asyncio.run(main.generate_reels(object(), direct_payload))
        assert direct_retry.status_code == 202
        direct_retry_job_id = json.loads(direct_retry.body)["job_id"]
        assert direct_retry_job_id != direct_job_id
        direct_retry_job = generation_jobs.get_job(conn, direct_retry_job_id)
        assert direct_retry_job
        assert not direct_retry_job["source_generation_id"]
        assert cross_request_lookup.call_count == 0

        repeated_direct_retry = asyncio.run(
            main.generate_reels(object(), direct_payload)
        )
        assert json.loads(repeated_direct_retry.body)["job_id"] == direct_retry_job_id
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs WHERE concept_id = 'c1'"
        ).fetchone()[0] == 2
        generation_jobs.transition_terminal(
            conn,
            job_id=direct_retry_job_id,
            status="failed",
            error_code="generation_failed",
            error_message="test recovery released the active queue slot",
        )

        cross_request_lookup.return_value = None
        first_feed = main.feed(object(), material_id="m1", autofill=True)
        feed_job_id = str(first_feed["generation_job_id"] or "")
        assert feed_job_id
        feed_job = generation_jobs.get_job(conn, feed_job_id)
        assert feed_job
        feed_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id=None,
            request_key=str(feed_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=feed_job_id,
            status="failed",
            result_generation_id=feed_generation_id,
            error_code="provider_quota_exhausted",
            usage={
                "counters": {"provider_cursor_open": 1},
                "consumed_video_ids": [],
            },
        )

        cross_request_lookup.reset_mock()
        wake_before_repeat = wake.call_count
        repeated_feed = main.feed(object(), material_id="m1", autofill=True)
        repeated_feed_again = main.feed(object(), material_id="m1", autofill=True)
        assert repeated_feed["generation_job_id"] == feed_job_id
        assert repeated_feed["generation_job_status"] == "failed"
        assert repeated_feed_again["generation_job_id"] == feed_job_id
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs WHERE concept_id IS NULL"
        ).fetchone()[0] == 1
        assert cross_request_lookup.call_count == 0
        assert wake.call_count == wake_before_repeat
    finally:
        conn.close()


@pytest.mark.parametrize(
    "terminal_error_code",
    ["provider_rate_limited", "provider_transient"],
)
def test_feed_suppresses_terminal_provider_failure_without_source_work(
    monkeypatch,
    terminal_error_code: str,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    wake = mock.Mock()
    cross_request_lookup = mock.Mock(return_value=None)
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    monkeypatch.setattr(main, "_ranked_request_reels", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        main,
        "_verified_cross_request_source_generation",
        cross_request_lookup,
    )
    try:
        first_feed = main.feed(object(), material_id="m1", autofill=True)
        job_id = str(first_feed["generation_job_id"] or "")
        job = generation_jobs.get_job(conn, job_id)
        assert job
        generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id=None,
            request_key=str(job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        conn.execute(
            "UPDATE reel_generation_jobs SET attempt_count = max_attempts WHERE id = ?",
            (job_id,),
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=job_id,
            status="failed",
            result_generation_id=generation_id,
            error_code=terminal_error_code,
            usage={
                "counters": {"provider_cursor_open": 0},
                "consumed_video_ids": [],
                "failed_source_attempts": {},
            },
        )

        cross_request_lookup.reset_mock()
        wake_before_repeat = wake.call_count
        repeated = main.feed(object(), material_id="m1", autofill=True)
        repeated_again = main.feed(object(), material_id="m1", autofill=True)

        assert repeated["generation_job_id"] == job_id
        assert repeated["generation_job_status"] == "failed"
        assert repeated_again["generation_job_id"] == job_id
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs WHERE concept_id IS NULL"
        ).fetchone()[0] == 1
        assert cross_request_lookup.call_count == 0
        assert wake.call_count == wake_before_repeat
    finally:
        conn.close()


def test_generate_exhausted_retry_stops_after_recovered_semantic_zero(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    root_payload = ReelsGenerateRequest(
        material_id="m1",
        concept_id="c1",
        num_reels=3,
    )
    failed_video_id = "BBBBBBBBBBB"
    try:
        root_response = asyncio.run(main.generate_reels(object(), root_payload))
        root_job_id = json.loads(root_response.body)["job_id"]
        root_job = generation_jobs.get_job(conn, root_job_id)
        assert root_job
        root_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(root_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=root_job_id,
            status="exhausted",
            result_generation_id=root_generation_id,
            usage={
                "counters": {"provider_cursor_open": 0},
                "consumed_video_ids": [],
                "failed_source_attempts": {failed_video_id: 1},
            },
        )

        continuation_payload = ReelsGenerateRequest(
            material_id="m1",
            concept_id="c1",
            num_reels=3,
            continuation_token=root_job_id,
        )
        retry = asyncio.run(main.generate_reels(object(), continuation_payload))
        assert retry.status_code == 202
        retry_job_id = json.loads(retry.body)["job_id"]
        retry_job = generation_jobs.get_job(conn, retry_job_id)
        assert retry_job
        assert retry_job["source_generation_id"] == root_generation_id

        retry_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(retry_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
            source_generation_id=root_generation_id,
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=retry_job_id,
            status="exhausted",
            result_generation_id=retry_generation_id,
            usage={
                "counters": {"provider_cursor_open": 0},
                "consumed_video_ids": [failed_video_id],
                "failed_source_attempts": {},
            },
        )

        terminal = asyncio.run(main.generate_reels(
            object(),
            ReelsGenerateRequest(
                material_id="m1",
                concept_id="c1",
                num_reels=3,
                continuation_token=retry_job_id,
            ),
        ))
        assert terminal["terminal_status"] == "exhausted"
        assert terminal["reels"] == []
        assert main._generation_chain_retryable_failed_source_ids(
            conn,
            generation_id=retry_generation_id,
        ) == set()
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_generate_continuation_survives_clip_concepts_created_by_the_previous_batch(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    root_payload = ReelsGenerateRequest(material_id="m1", num_reels=3)
    try:
        root_response = asyncio.run(main.generate_reels(object(), root_payload))
        root_job_id = json.loads(root_response.body)["job_id"]
        root_job = generation_jobs.get_job(conn, root_job_id)
        assert root_job
        root_fingerprint = str(root_job["content_fingerprint"])
        root_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id=None,
            request_key=str(root_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        ensure_clip_concept(
            conn,
            material_id="m1",
            title="worked ATP example",
            semantic_identity="ATP synthesis",
        )
        assert generation_jobs.material_content_fingerprint(conn, "m1") == root_fingerprint
        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'partial', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (root_generation_id, completed_at, completed_at, root_job_id),
        )

        continuation = asyncio.run(main.generate_reels(
            object(),
            ReelsGenerateRequest(
                material_id="m1",
                num_reels=3,
                continuation_token=root_job_id,
            ),
        ))

        assert continuation.status_code == 202
        continuation_job_id = json.loads(continuation.body)["job_id"]
        continuation_job = generation_jobs.get_job(conn, continuation_job_id)
        assert continuation_job
        assert main._job_request_params(continuation_job)["continuation_token"] == root_job_id
    finally:
        conn.close()


@pytest.mark.parametrize("stale_dimension", ["adaptation", "request_schema"])
def test_generate_rejects_a_stale_continuation(monkeypatch, stale_dimension: str) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    root_payload = ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=3)
    try:
        root_response = asyncio.run(main.generate_reels(object(), root_payload))
        root_job_id = json.loads(root_response.body)["job_id"]
        root_job = generation_jobs.get_job(conn, root_job_id)
        assert root_job
        root_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(root_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'completed', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (root_generation_id, completed_at, completed_at, root_job_id),
        )

        if stale_dimension == "adaptation":
            monkeypatch.setattr(
                main,
                "_learner_concept_signals",
                lambda *_args, **_kwargs: {
                    "mitochondria": {
                        "helpful": 0.0,
                        "confusing": 1.0,
                        "adjustment": -0.35,
                    }
                },
            )
        else:
            stored_params = main._job_request_params(root_job)
            stored_params["request_schema_version"] = "quality_silence_v38"
            conn.execute(
                "UPDATE reel_generation_jobs SET request_params_json = ? WHERE id = ?",
                (json.dumps(stored_params), root_job_id),
            )

        with pytest.raises(main.HTTPException) as exc_info:
            asyncio.run(main.generate_reels(
                object(),
                ReelsGenerateRequest(
                    material_id="m1",
                    concept_id="c1",
                    num_reels=3,
                    continuation_token=root_job_id,
                ),
            ))

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "invalid_continuation_token"
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 1
    finally:
        conn.close()


def test_generate_cancels_a_stale_active_job_before_queueing_its_replacement(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    payload = ReelsGenerateRequest(material_id="m1", concept_id="c1", num_reels=3)
    try:
        first = asyncio.run(main.generate_reels(object(), payload))
        first_job_id = json.loads(first.body)["job_id"]
        monkeypatch.setattr(
            main,
            "_learner_concept_signals",
            lambda *_args, **_kwargs: {
                "mitochondria": {
                    "helpful": 0.0,
                    "confusing": 1.0,
                    "adjustment": -0.35,
                }
            },
        )

        replacement = asyncio.run(main.generate_reels(object(), payload))
        replacement_job_id = json.loads(replacement.body)["job_id"]

        assert replacement.status_code == 202
        assert replacement_job_id != first_job_id
        assert generation_jobs.get_job(conn, first_job_id)["status"] == "cancelled"
        assert generation_jobs.get_job(conn, replacement_job_id)["status"] == "queued"
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs "
            "WHERE status IN ('queued', 'running')"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_feedback_cancels_the_previous_adaptation_inside_its_write(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="feedback-signal-generation",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=generation_id,
        reel_id="feedback-signal-reel",
        video_id="feedback-signal-video",
        created_at=now.isoformat(),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="feedback-signal-job",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="feedback-signal-worker",
        now=now,
    )
    assert leased
    signal_written = False

    def record_feedback(*_args, **_kwargs) -> int:
        nonlocal signal_written
        signal_written = True
        return 1

    def current_fingerprint(*_args, **_kwargs) -> str:
        assert signal_written
        return "after-feedback"

    monkeypatch.setattr(main.reel_service, "record_feedback", record_feedback)
    monkeypatch.setattr(
        main,
        "_learner_adaptation_fingerprint",
        current_fingerprint,
    )
    try:
        response = main.feedback(
            object(),
            FeedbackRequest(
                reel_id="feedback-signal-reel",
                helpful=True,
                confusing=False,
            ),
        )

        assert response["status"] == "ok"
        assert generation_jobs.get_job(conn, job["id"])["status"] == "cancelled"
    finally:
        conn.close()


def test_completed_assessment_cancels_the_previous_adaptation_inside_its_write(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    conn.execute(
        "INSERT INTO assessment_sessions "
        "(id, learner_id, material_id, status, created_at, updated_at) "
        "VALUES ('assessment-session', 'learner-1', 'm1', 'completed', ?, ?)",
        (now.isoformat(), now.isoformat()),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="assessment-signal-job",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="assessment-signal-worker",
        now=now,
    )
    assert leased
    adaptation_locked = False
    signal_written = False
    original_lock = main._lock_learner_adaptation

    def lock_adaptation(worker_conn, **kwargs) -> None:
        nonlocal adaptation_locked
        original_lock(worker_conn, **kwargs)
        adaptation_locked = True

    def answer(*_args, **_kwargs):
        nonlocal signal_written
        assert adaptation_locked
        signal_written = True
        return {
            "correct": False,
            "correct_index": 0,
            "explanation": "Review the concept.",
            "session": {"status": "completed", "material_id": "m1"},
        }

    def current_fingerprint(*_args, **_kwargs) -> str:
        assert signal_written
        return "after-assessment"

    monkeypatch.setattr(main, "_lock_learner_adaptation", lock_adaptation)
    monkeypatch.setattr(main.assessment_service, "answer", answer)
    monkeypatch.setattr(main.reel_service, "update_level_adjustment", mock.Mock())
    monkeypatch.setattr(
        main,
        "_learner_adaptation_fingerprint",
        current_fingerprint,
    )
    try:
        response = main.answer_assessment(
            object(),
            "assessment-session",
            AssessmentAnswerRequest(question_id="assessment-question", choice_index=1),
        )

        assert response["session"]["status"] == "completed"
        assert generation_jobs.get_job(conn, job["id"])["status"] == "cancelled"
    finally:
        conn.close()


def test_level_change_cancels_an_active_job_from_the_previous_difficulty(
    monkeypatch,
) -> None:
    conn = _conn()

    @contextmanager
    def connection(**_kwargs):
        yield conn

    monkeypatch.setattr(main, "get_conn", connection)
    monkeypatch.setattr(
        main,
        "_resolve_learner_identity",
        lambda *_args, **_kwargs: "learner-1",
    )
    main.reel_service.learner_progress(conn, "m1", "learner-1")
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="previous-level-job",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "knowledge_level": "beginner",
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
        },
        now=now,
    )
    try:
        response = main.update_material_level(
            "m1",
            object(),
            MaterialLevelUpdateRequest(knowledge_level="advanced"),
        )

        assert response["knowledge_level"] == "advanced"
        assert generation_jobs.get_job(conn, job["id"])["status"] == "cancelled"
        progress = main.reel_service.learner_progress(conn, "m1", "learner-1")
        assert progress["selected_level"] == "advanced"
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
                "selection_contract_version": "quality_silence_v39",
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
        assert [event["type"] for event in events] == ["final", "terminal"]
        assert [event["seq"] for event in events] == [3, 4]
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
                "selection_contract_version": "quality_silence_v39",
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
            "selection_contract_version": "quality_silence_v39",
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
                "selection_contract_version": "quality_silence_v39",
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
    ] == []
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
def test_boundary_only_failure_surfaces_strict_transcript_fallback_in_feed_and_replay(
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
    assert [item["reel_id"] for item in ranked_feed] == [persisted.reel_id]

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
    assert [item["reel_id"] for item in status["reels"]] == [persisted.reel_id]
    assert [
        event["payload"]["reel"]["reel_id"]
        for event in replay
        if event["type"] == "candidate"
    ] == []
    final = next(event for event in replay if event["type"] == "final")
    assert final["payload"]["authoritative"] is True
    assert [item["reel_id"] for item in final["payload"]["reels"]] == [
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


def test_preflight_is_cache_only_and_does_not_generate(monkeypatch) -> None:
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
        assert result["availability"] == "unknown"
        assert result["candidate_count"] == 0
        assert provider_calls == []
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
        queued_job = generation_jobs.get_job(conn, queued["generation_job_id"])
        assert queued_job is not None
        assert json.loads(queued_job["request_params_json"])["num_reels"] == 9
        assert main.GENERATION_OUTPUT_CEILINGS == {"fast": 9, "slow": 9}
        wake.assert_called_once_with()
        assert rate_calls == [
            ("feed", 36),
            ("feed", 36),
            ("generation-submit", 6),
        ]
    finally:
        conn.close()


def test_feed_cancels_a_stale_active_job_before_queueing_its_replacement(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    concept_signals: dict[str, dict[str, float]] = {}
    monkeypatch.setattr(
        main,
        "_learner_concept_signals",
        lambda *_args, **_kwargs: concept_signals,
    )
    try:
        first = main.feed(object(), material_id="m1", autofill=True)
        first_job_id = first["generation_job_id"]
        assert first_job_id

        concept_signals = {
            "c1": {"helpful": 0.0, "confusing": 1.0, "adjustment": -0.35}
        }
        replacement = main.feed(object(), material_id="m1", autofill=True)
        replacement_job_id = replacement["generation_job_id"]

        assert replacement_job_id
        assert replacement_job_id != first_job_id
        assert generation_jobs.get_job(conn, first_job_id)["status"] == "cancelled"
        assert generation_jobs.get_job(conn, replacement_job_id)["status"] == "queued"
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs "
            "WHERE status IN ('queued', 'running')"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_feed_does_not_top_up_a_full_three_batch_startup_reservoir(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(
        main,
        "_fetch_active_generation_row",
        lambda *_args, **_kwargs: {"id": "nine-ready-generation"},
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": f"ready-{index}",
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
            }
            for index in range(9)
        ],
    )
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    try:
        response = main.feed(
            object(),
            material_id="m1",
            page=1,
            limit=5,
            prefetch=9,
            autofill=True,
        )

        assert len(response["reels"]) == 5
        assert response["generation_job_id"] is None
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 0
        wake.assert_not_called()
    finally:
        conn.close()


def test_feed_drops_stale_ranked_rows_before_returning_current_inventory(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(
        main,
        "_fetch_active_generation_row",
        lambda *_args, **_kwargs: {"id": "current-generation"},
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": "stale-v36",
                "selection_contract_version": "quality_silence_v36",
            },
            {
                "reel_id": "current-v37",
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
            },
        ],
    )
    try:
        response = main.feed(object(), material_id="m1", autofill=False)

        assert [reel["reel_id"] for reel in response["reels"]] == ["current-v37"]
        assert response["total"] == 1
    finally:
        conn.close()


def test_feed_page_two_never_starts_provider_generation(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(
        main,
        "_fetch_active_generation_row",
        lambda *_args, **_kwargs: {"id": "partial-ready-generation"},
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": f"ready-{index}",
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
            }
            for index in range(5)
        ],
    )
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    try:
        response = main.feed(
            object(),
            material_id="m1",
            page=2,
            limit=3,
            prefetch=9,
            autofill=True,
        )

        assert len(response["reels"]) == 2
        assert response["generation_job_id"] is None
        assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 0
        wake.assert_not_called()
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
        assert response["continuation_token"] == terminal["id"]
        wake.assert_not_called()
    finally:
        conn.close()


def test_feed_reload_reports_the_latest_continuation_batch(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    ranked_kwargs: dict[str, object] = {}

    def ranked(*_args, **kwargs):
        ranked_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(main, "_ranked_request_reels", ranked)
    try:
        queued = main.feed(object(), material_id="m1", autofill=True)
        root_job = generation_jobs.get_job(conn, queued["generation_job_id"])
        assert root_job
        root_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id=None,
            request_key=str(root_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'completed', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (root_generation_id, completed_at, completed_at, root_job["id"]),
        )
        root_params = json.loads(root_job["request_params_json"])
        continuation_job, _ = generation_jobs.submit_or_get_active(
            conn,
            material_id="m1",
            concept_id=None,
            request_key="feed-continuation-request",
            content_fingerprint=str(root_job["content_fingerprint"]),
            learner_id="learner-1",
            request_params={
                **root_params,
                "continuation_token": root_job["id"],
            },
            source_generation_id=root_generation_id,
            now=datetime.now(timezone.utc) + timedelta(seconds=1),
        )
        continuation_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id=None,
            request_key="feed-continuation-request",
            generation_mode="slow",
            retrieval_profile="unified",
            source_generation_id=root_generation_id,
        )
        later = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'partial', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (
                continuation_generation_id,
                later,
                later,
                continuation_job["id"],
            ),
        )

        response = main.feed(object(), material_id="m1", autofill=False)

        assert response["generation_job_id"] == continuation_job["id"]
        assert response["generation_job_status"] == "partial"
        assert response["continuation_token"] == continuation_job["id"]
        assert response["generation_id"] == continuation_generation_id
        assert ranked_kwargs["released_only"] is True
    finally:
        conn.close()


def test_feed_reload_queues_bounded_retry_from_failed_source_state(monkeypatch) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    monkeypatch.setattr(main, "_ranked_request_reels", lambda *_args, **_kwargs: [])
    try:
        first = main.feed(object(), material_id="m1", autofill=True)
        first_job = generation_jobs.get_job(conn, first["generation_job_id"])
        assert first_job
        first_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id=None,
            request_key=str(first_job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        generation_jobs.transition_terminal(
            conn,
            job_id=str(first_job["id"]),
            status="failed",
            result_generation_id=first_generation_id,
            usage={
                "counters": {"provider_cursor_open": 0},
                "consumed_video_ids": [],
                "failed_source_attempts": {"CCCCCCCCCCC": 1},
            },
        )

        recovered = main.feed(object(), material_id="m1", autofill=True)

        assert recovered["generation_job_status"] == "queued"
        assert recovered["generation_job_id"] != first_job["id"]
        retry_job = generation_jobs.get_job(conn, recovered["generation_job_id"])
        assert retry_job
        assert retry_job["source_generation_id"] == first_generation_id
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 2
        assert wake.call_count == 2
    finally:
        conn.close()


def test_feed_ignores_prior_contract_job_and_starts_current_generation(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    content_fingerprint = generation_jobs.material_content_fingerprint(conn, "m1", None)
    stale_job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="quality-silence-v36-request",
        content_fingerprint=content_fingerprint,
        learner_id="learner-1",
        request_params={
            "knowledge_level": "beginner",
            "generation_mode": "slow",
            "creative_commons_only": False,
            "preferred_video_duration": "any",
            "language": "en",
            "exclude_video_ids": [],
            "min_relevance": None,
        },
    )
    stale_params = json.loads(stale_job["request_params_json"])
    stale_params["request_schema_version"] = "quality_silence_v36"
    conn.execute(
        "UPDATE reel_generation_jobs SET status = 'completed', "
        "result_generation_id = ?, request_params_json = ? WHERE id = ?",
        ("stale-v36-generation", json.dumps(stale_params), stale_job["id"]),
    )

    response = main.feed(object(), material_id="m1", autofill=True)

    assert response["reels"] == []
    assert response["generation_job_id"] != stale_job["id"]
    assert response["generation_job_status"] == "queued"
    current_job = generation_jobs.get_job(conn, response["generation_job_id"])
    assert current_job is not None
    assert json.loads(current_job["request_params_json"])[
        "request_schema_version"
    ] == generation_jobs.REQUEST_SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM reel_generation_jobs").fetchone()[0] == 2
    wake.assert_called_once_with()
    conn.close()


def test_feed_paginates_five_clip_organizer_release_from_three_clip_request(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    monkeypatch.setattr(main, "_wake_generation_worker", mock.Mock())
    try:
        queued = main.feed(
            object(),
            material_id="m1",
            page=1,
            limit=3,
            prefetch=3,
            autofill=True,
        )
        job = generation_jobs.get_job(conn, queued["generation_job_id"])
        assert job
        assert json.loads(job["request_params_json"])["num_reels"] == 3
        generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id=None,
            request_key=str(job["request_key"]),
            generation_mode="slow",
            retrieval_profile="unified",
        )
        created_at = datetime.now(timezone.utc)
        for index in range(6):
            _insert_generation_reel(
                conn,
                generation_id=generation_id,
                reel_id=f"paged-reel-{index}",
                video_id=f"paged-video-{index}",
                created_at=(created_at + timedelta(seconds=index)).isoformat(),
            )
        main._complete_generation(
            conn,
            generation_id=generation_id,
            retrieval_profile="unified",
            status="completed",
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'completed', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (generation_id, completed_at, completed_at, job["id"]),
        )
        _append_authoritative_release(
            conn,
            job_id=str(job["id"]),
            reel_ids=[f"paged-reel-{index}" for index in range(5)],
        )
        ranked_rows = [
            {
                "reel_id": f"paged-reel-{index}",
                "video_id": f"paged-video-{index}",
                "t_start": 0.0,
                "t_end": 30.0,
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
                "_selection_ordered": True,
            }
            for index in range(6)
        ]
        monkeypatch.setattr(
            main.reel_service,
            "ranked_feed",
            lambda *_args, **_kwargs: [dict(row) for row in ranked_rows],
        )
        monkeypatch.setattr(
            main,
            "_shape_request_page_reels",
            lambda rows, **_kwargs: list(rows),
        )
        monkeypatch.setattr(
            main,
            "_finalize_request_reel_order",
            lambda _conn, **kwargs: kwargs["rows"],
        )

        first = main.feed(object(), material_id="m1", page=1, limit=3, autofill=False)
        normal_second = main.feed(
            object(), material_id="m1", page=2, limit=3, autofill=False
        )
        first_ids = {reel["reel_id"] for reel in first["reels"]}
        normal_second_ids = {reel["reel_id"] for reel in normal_second["reels"]}
        assert len(first_ids) == 3
        assert len(normal_second_ids) == 2
        assert first_ids.isdisjoint(normal_second_ids)
        assert "paged-reel-5" not in first_ids | normal_second_ids

        for reel_id in first_ids:
            main.assessment_service.record_scroll(
                conn,
                learner_id="learner-1",
                reel_id=reel_id,
            )

        continued_second = main.feed(
            object(), material_id="m1", page=2, limit=3, autofill=False
        )
        continued_ids = {reel["reel_id"] for reel in continued_second["reels"]}
        assert continued_ids == normal_second_ids
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
            {
                "reel_id": f"ranked-{index}",
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
            }
            for index in range(5)
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


@pytest.mark.parametrize(
    ("prior_mode", "prior_relevance", "current_mode", "current_relevance"),
    [
        ("slow", None, "fast", None),
        ("fast", 0.3, "fast", 0.0),
    ],
)
def test_feed_verified_reservoir_satisfies_compatible_request_without_queuing(
    monkeypatch,
    prior_mode: str,
    prior_relevance: float | None,
    current_mode: str,
    current_relevance: float | None,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    completed_at = datetime.now(timezone.utc).isoformat()
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="prior-level-request",
        generation_mode=prior_mode,
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
            "generation_mode": prior_mode,
            "num_reels": 12,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": prior_relevance,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
            "language": "en",
        },
    )
    ranked_generation_ids: list[str | None] = []
    ranked_relevance_thresholds: list[float | None] = []

    def ranked(*_args, **kwargs):
        ranked_generation_ids.append(kwargs.get("generation_id"))
        ranked_relevance_thresholds.append(kwargs.get("min_relevance"))
        return [
            {
                "reel_id": f"verified-prior-reel-{index}",
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
            }
            for index in range(12)
        ]

    monkeypatch.setattr(main, "_ranked_request_reels", ranked)
    try:
        response = main.feed(
            object(),
            material_id="m1",
            autofill=True,
            generation_mode=current_mode,
            min_relevance=current_relevance,
        )

        assert response["generation_id"] == source_generation_id
        assert len(response["reels"]) == 5
        assert ranked_generation_ids == [source_generation_id]
        assert ranked_relevance_thresholds == [current_relevance]
        queued = conn.execute(
            "SELECT * FROM reel_generation_jobs WHERE status = 'queued' "
            "ORDER BY created_at DESC, id DESC LIMIT 1"
        ).fetchone()
        assert queued is None
    finally:
        conn.close()


def test_cross_request_reuse_prefers_exact_relevance_before_fallback() -> None:
    conn = _conn()
    completed_at = datetime.now(timezone.utc)
    base_params = {
        "generation_mode": "fast",
        "num_reels": 9,
        "exclude_video_ids": [],
        "creative_commons_only": False,
        "preferred_video_duration": "any",
        "knowledge_level": "beginner",
        "language": "en",
    }
    generation_ids: dict[float, str] = {}
    for index, relevance in enumerate((0.0, 0.9)):
        generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=f"relevance-{relevance}",
            generation_mode="fast",
            retrieval_profile="unified",
        )
        generation_ids[relevance] = generation_id
        timestamp = (completed_at + timedelta(seconds=index)).isoformat()
        _insert_generation_reel(
            conn,
            generation_id=generation_id,
            reel_id=f"relevance-{relevance}-reel",
            video_id=f"relevance-{relevance}-video",
            created_at=timestamp,
        )
        _terminal_job_for_generation(
            conn,
            request_key=f"relevance-{relevance}",
            generation_id=generation_id,
            completed_at=timestamp,
            content_fingerprint="same-fingerprint",
            request_params={**base_params, "min_relevance": relevance},
        )
    try:
        result = main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="new-request",
            concept_id="c1",
            content_fingerprint="same-fingerprint",
            request_params={**base_params, "min_relevance": 0.0},
        )
        assert result == generation_ids[0.0]
    finally:
        conn.close()


def test_cross_request_reuse_requires_current_schema_and_adaptation() -> None:
    conn = _conn()
    params = {
        "generation_mode": "fast",
        "num_reels": 9,
        "exclude_video_ids": [],
        "creative_commons_only": False,
        "preferred_video_duration": "any",
        "knowledge_level": "beginner",
        "language": "en",
        "min_relevance": 0.0,
    }
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="legacy-conceptless-request",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    _insert_generation_reel(
        conn,
        generation_id=generation_id,
        reel_id="legacy-conceptless-reel",
        video_id="legacy-conceptless-video",
        created_at="2026-07-12T00:00:00+00:00",
    )
    job = _terminal_job_for_generation(
        conn,
        request_key="legacy-conceptless-request",
        generation_id=generation_id,
        completed_at="2026-07-12T00:00:00+00:00",
        content_fingerprint="same-fingerprint",
        request_params=params,
    )
    stored = main._job_request_params(job)
    stored["request_schema_version"] = "quality_silence_v38"
    conn.execute(
        "UPDATE reel_generation_jobs SET request_params_json = ? WHERE id = ?",
        (json.dumps(stored), job["id"]),
    )
    try:
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="current-request",
            concept_id="c1",
            content_fingerprint="same-fingerprint",
            request_params=params,
        ) is None

        stored.update({
            "request_schema_version": main.GENERATION_REQUEST_SCHEMA_VERSION,
            "adaptation_fingerprint": "before-feedback",
        })
        conn.execute(
            "UPDATE reel_generation_jobs SET request_params_json = ? WHERE id = ?",
            (json.dumps(stored), job["id"]),
        )
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="after-feedback-request",
            concept_id="c1",
            content_fingerprint="same-fingerprint",
            request_params={**params, "adaptation_fingerprint": "after-feedback"},
        ) is None
        assert main._verified_cross_request_source_generation(
            conn,
            material_id="m1",
            learner_id="learner-1",
            request_key="matching-feedback-request",
            concept_id="c1",
            content_fingerprint="same-fingerprint",
            request_params={**params, "adaptation_fingerprint": "before-feedback"},
        ) == generation_id
    finally:
        conn.close()


def test_feed_partial_cross_request_reservoir_queues_fresh_startup_budget(
    monkeypatch,
) -> None:
    conn = _conn()
    _patch_request_context(monkeypatch, conn)
    completed_at = datetime.now(timezone.utc).isoformat()
    source_generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="prior-partial-startup-request",
        generation_mode="slow",
        retrieval_profile="unified",
    )
    for index in range(3):
        _insert_generation_reel(
            conn,
            generation_id=source_generation_id,
            reel_id=f"strict-cached-reel-{index}",
            video_id=f"strict-cached-video-{index}",
            created_at=completed_at,
        )
    _terminal_job_for_generation(
        conn,
        request_key="prior-partial-startup-request",
        generation_id=source_generation_id,
        completed_at=completed_at,
        concept_id=None,
        content_fingerprint=generation_jobs.material_content_fingerprint(
            conn, "m1", None
        ),
        request_params={
            "generation_mode": "slow",
            "num_reels": 9,
            "exclude_video_ids": [],
            "creative_commons_only": False,
            "min_relevance": None,
            "preferred_video_duration": "any",
            "knowledge_level": "beginner",
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
            "language": "en",
        },
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": f"strict-cached-reel-{index}",
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
            }
            for index in range(3)
        ],
    )
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    try:
        response = main.feed(
            object(),
            material_id="m1",
            limit=9,
            prefetch=9,
            autofill=True,
            generation_mode="fast",
        )

        assert [reel["reel_id"] for reel in response["reels"]] == [
            f"strict-cached-reel-{index}" for index in range(3)
        ]
        assert response["generation_id"] == source_generation_id
        queued = generation_jobs.get_job(conn, response["generation_job_id"])
        assert queued is not None
        assert queued["source_generation_id"] == source_generation_id
        queued_params = main._job_request_params(queued)
        assert queued_params["num_reels"] == 9
        assert queued_params["fresh_source_budget"] is True
        wake.assert_called_once_with()
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
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
            "language": "en",
        },
    )
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: [
            {
                "reel_id": f"prior-fast-feed-reel-{index}",
                "selection_contract_version": main.SELECTION_CONTRACT_VERSION,
            }
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


def test_feed_queues_when_cross_request_reservoir_has_no_reusable_clip(
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
        # Relevance is reapplied while ranking the reused chain. It must not
        # force duplicate provider work or duplicate persistence by itself.
        ("same-fingerprint", {"min_relevance": 0.9}, True),
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
            "selection_contract_version": "quality_silence_v39",
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
            "selection_contract_version": "quality_silence_v39",
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

        assert [item["reel_id"] for item in ranked] == [
            "intermediate-only",
            "advanced-only",
        ]
    finally:
        conn.close()


def test_cross_level_reservoir_ignores_invalid_rows_when_valid_inventory_remains() -> None:
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
        ) == child_generation_id

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
        ) == child_generation_id

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
        ) == child_generation_id
    finally:
        conn.close()


def test_generate_cross_request_reservoir_is_owned_by_current_batch(
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
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
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
                "selection_contract_version": "quality_silence_v39",
            }
        ],
    )
    wake = mock.Mock()
    monkeypatch.setattr(main, "_wake_generation_worker", wake)
    try:
        payload = ReelsGenerateRequest(
            material_id="m1",
            concept_id="c1",
            generation_mode="fast",
            num_reels=5,
        )
        queued_response = asyncio.run(main.generate_reels(
            object(),
            payload,
        ))

        assert isinstance(queued_response, JSONResponse)
        assert queued_response.status_code == 202
        current_job_id = json.loads(queued_response.body)["job_id"]
        current_job = generation_jobs.get_job(conn, current_job_id)
        assert current_job is not None
        assert current_job["source_generation_id"] == source_generation_id
        assert main._job_request_params(current_job)["fresh_source_budget"] is True

        current_generation_id = main._create_generation_row(
            conn,
            material_id="m1",
            concept_id="c1",
            request_key=str(current_job["request_key"]),
            generation_mode="fast",
            retrieval_profile="unified",
            source_generation_id=source_generation_id,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE reel_generation_jobs SET status = 'completed', phase = 'terminal', "
            "progress = 1.0, result_generation_id = ?, completed_at = ?, updated_at = ? "
            "WHERE id = ?",
            (current_generation_id, completed_at, completed_at, current_job_id),
        )

        response = asyncio.run(main.generate_reels(object(), payload))

        assert response["generation_id"] == current_generation_id
        assert response["batch_id"] == current_job_id
        assert response["batch_size"] == 1
        assert response["continuation_token"] == current_job_id
        assert response["terminal_status"] == "completed"
        assert response["reels"] == [
            {
                "reel_id": "verified-concept-reel",
                "video_id": "verified-concept-video-0",
                "selection_contract_version": "quality_silence_v39",
            }
        ]
        assert conn.execute(
            "SELECT COUNT(*) FROM reel_generation_jobs"
        ).fetchone()[0] == 2
        wake.assert_called_once_with()
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
            "adaptation_fingerprint": EMPTY_ADAPTATION_FINGERPRINT,
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


def test_generate_queues_when_cross_request_reservoir_has_no_reusable_clip(
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
