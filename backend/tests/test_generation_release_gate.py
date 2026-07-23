from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from backend.app import main
from backend.app.clip_engine.provider_runtime import ProviderUsageRecord
from backend.app.services import generation_jobs
from backend.tests import test_generation_job_api as job_api


def test_worker_releases_only_final_ranked_inventory(monkeypatch) -> None:
    conn = job_api._conn()
    job_api._patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="final-ranking-release-gate",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 3,
            "knowledge_level": "beginner",
        },
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="release-gate-worker",
        now=now,
    )
    assert leased
    generated: list[dict] = []

    def generate_stage(worker_conn, **kwargs) -> None:
        for index in range(3):
            reel = job_api._insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id=f"release-reel-{index}",
                video_id=f"release-video-{index}",
                created_at=(now + timedelta(seconds=index)).isoformat(),
            )
            generated.append(reel)
            kwargs["on_reel_created"](reel)
            assert [
                event["payload"]["reel"]["reel_id"]
                for event in generation_jobs.replay_events(
                    conn,
                    job_id=job["id"],
                )
                if event["type"] == "candidate"
            ] == [
                f"release-reel-{emitted_index}"
                for emitted_index in range(index + 1)
            ]

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(reversed(generated)),
    )
    try:
        main._run_leased_generation_job(leased, threading.Event())

        expected_ids = ["release-reel-2", "release-reel-1", "release-reel-0"]
        events = generation_jobs.replay_events(conn, job_id=job["id"])
        assert [event["type"] for event in events] == [
            "candidate",
            "candidate",
            "candidate",
            "final",
            "terminal",
        ]
        final = next(event for event in events if event["type"] == "final")
        assert [reel["reel_id"] for reel in final["payload"]["reels"]] == expected_ids
    finally:
        conn.close()


def test_terminal_cost_uses_ten_authoritative_reels_after_nine_provisionals(
    monkeypatch,
) -> None:
    conn = job_api._conn()
    job_api._patch_transactional_worker_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id="c1",
        request_key="authoritative-release-cost-denominator",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={
            "generation_mode": "fast",
            "num_reels": 9,
            "knowledge_level": "beginner",
        },
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="authoritative-release-cost-worker",
        now=now,
    )
    assert leased
    generated: list[dict] = []
    retrieval_calls = 0
    release_attempts = 0
    original_terminal = main.transition_generation_terminal

    def generate_stage(worker_conn, **kwargs) -> None:
        nonlocal retrieval_calls
        retrieval_calls += 1
        context = kwargs["generation_context"]
        context.increment_counter("persisted_clips", 9)
        context.record(ProviderUsageRecord(
            provider="groq",
            operation="transcript",
            attempt=1,
            timestamp=now.isoformat(),
            status_code=200,
            billable_requests=1,
            model_used="whisper-large-v3-turbo",
            metadata={
                "provider_call": True,
                "stage": "groq_boundary_asr",
                "physical_dispatches": 1,
                "groq_dispatch_id": "authoritative-release-cost-dispatch",
                "billing_usage_known": True,
                "billing_unknown_attempts": 0,
                "billing_unknown_reserved_cost_usd": 0.0,
                "actual_cost_usd": 1.0,
                "audio_seconds": 10.0,
                "billed_audio_seconds": 10.0,
            },
        ))
        for index in range(10):
            reel = job_api._insert_generation_reel(
                worker_conn,
                generation_id=str(kwargs["generation_id"]),
                reel_id=f"authoritative-release-reel-{index}",
                video_id=f"AUTHORIT{index:02d}",
                created_at=(now + timedelta(seconds=index)).isoformat(),
            )
            generated.append(reel)
            if index < 9:
                kwargs["on_reel_created"](reel)

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

    def transient_release(worker_conn, **kwargs):
        nonlocal release_attempts
        if kwargs.get("status") == "completed":
            release_attempts += 1
            if release_attempts == 1:
                raise job_api._PostgresTransactionFailure("40P01")
        return original_terminal(worker_conn, **kwargs)

    monkeypatch.setattr(main.reel_service, "generate_reels", generate_stage)
    monkeypatch.setattr(
        main,
        "_ranked_request_reels",
        lambda *_args, **_kwargs: list(generated),
    )
    monkeypatch.setattr(main, "order_lesson_batch", order_batch)
    monkeypatch.setattr(main, "transition_generation_terminal", transient_release)
    try:
        main._run_leased_generation_job(leased, threading.Event())

        completed = generation_jobs.get_job(conn, job["id"])
        assert completed and completed["status"] == "completed"
        assert retrieval_calls == 1
        assert release_attempts == 2

        events = generation_jobs.replay_events(conn, job_id=job["id"])
        assert len([
            event for event in events if event["type"] == "candidate"
        ]) == 9
        final = next(event for event in events if event["type"] == "final")
        assert len(final["payload"]["reels"]) == 10

        stored_usage = json.loads(str(completed["usage_json"] or "{}"))
        status_usage = main._generation_job_status_payload(
            conn,
            completed,
        )["usage"]
        replay_usage = next(
            event
            for event in main._sanitize_generation_replay_events(
                conn,
                completed,
                events,
            )
            if event["type"] == "terminal"
        )["payload"]["usage"]
        for usage in (stored_usage, status_usage, replay_usage):
            summary = usage["summary"]
            assert summary["accepted_clips"] == 9
            assert summary["released_reels"] == 10
            assert summary["cost_per_accepted_clip_usd"] == pytest.approx(
                1.0 / 9.0,
            )
            assert summary["cost_per_released_reel_usd"] == pytest.approx(
                0.1,
            )
    finally:
        conn.close()


def test_feed_hides_a_running_jobs_unranked_result_generation(monkeypatch) -> None:
    conn = job_api._conn()
    job_api._patch_request_context(monkeypatch, conn)
    now = datetime.now(timezone.utc)
    generation_id = main._create_generation_row(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="running-unranked-feed",
        generation_mode="fast",
        retrieval_profile="unified",
    )
    persisted = job_api._insert_generation_reel(
        conn,
        generation_id=generation_id,
        reel_id="running-unranked-reel",
        video_id="running-unranked-video",
        created_at=now.isoformat(),
    )
    job, _ = generation_jobs.submit_or_get_active(
        conn,
        material_id="m1",
        concept_id=None,
        request_key="running-unranked-feed",
        content_fingerprint="fingerprint",
        learner_id="learner-1",
        request_params={"generation_mode": "fast", "num_reels": 3},
        now=now,
    )
    leased = generation_jobs.lease_job(
        conn,
        job_id=job["id"],
        lease_owner="running-unranked-worker",
        now=now,
    )
    assert leased
    conn.execute(
        "UPDATE reel_generation_jobs SET result_generation_id = ? WHERE id = ?",
        (generation_id, job["id"]),
    )
    running = generation_jobs.get_job(conn, job["id"])
    assert running
    monkeypatch.setattr(main, "find_completed_generation_job", lambda *_args: None)
    monkeypatch.setattr(
        main,
        "find_active_generation_job",
        lambda *_args, **_kwargs: running,
    )
    monkeypatch.setattr(
        main,
        "_latest_compatible_generation_job",
        lambda *_args, **_kwargs: running,
    )
    ranked_feed = mock.Mock(return_value=[persisted])
    monkeypatch.setattr(main.reel_service, "ranked_feed", ranked_feed)

    try:
        response = main.feed(
            object(),
            material_id="m1",
            limit=3,
            prefetch=0,
            autofill=False,
            generation_mode="fast",
        )

        assert response["reels"] == []
        assert response["generation_id"] is None
        assert response["generation_job_id"] == job["id"]
        assert response["generation_job_status"] == "running"
        ranked_feed.assert_not_called()
    finally:
        conn.close()
