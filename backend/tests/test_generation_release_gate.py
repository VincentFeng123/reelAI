from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest import mock

from backend.app import main
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
