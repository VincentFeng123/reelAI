"""FastAPI assessment contract, privacy, resume, and learner isolation."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from contextlib import contextmanager
from unittest import mock

from fastapi.testclient import TestClient
import pytest

from backend.app import db as db_module
from backend.app.config import get_settings
import backend.app.main as main_module
from backend.app.services.assessments import store_reel_assessment_question


OWNER_HEADER = "x-studyreels-owner-key"
OWNER_A = "assessment-owner-a-abcdefghijklmnopqrstuvwxyz"
OWNER_B = "assessment-owner-b-abcdefghijklmnopqrstuvwxyz"
MATERIAL = "assessment-api-material"


class _PostgresFailure(Exception):
    def __init__(self, sqlstate: str, message: str = "postgres failure") -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


def _fail_first_commit_acknowledgement(real_get_conn):
    transaction_attempts: list[int] = []

    @contextmanager
    def connection(*, transactional: bool = False):
        with real_get_conn(transactional=transactional) as conn:
            yield conn
        if transactional:
            transaction_attempts.append(1)
            if len(transaction_attempts) == 1:
                raise _PostgresFailure("08006", "connection lost after commit")

    return connection, transaction_attempts


def test_adaptive_mutation_retry_declines_permanent_postgres_error(
    monkeypatch,
) -> None:
    attempts = 0

    @contextmanager
    def connection(*, transactional: bool = False):
        nonlocal attempts
        assert transactional is True
        attempts += 1
        raise _PostgresFailure("23505", "unique violation")
        yield None

    monkeypatch.setattr(main_module, "get_conn", connection)
    with pytest.raises(_PostgresFailure, match="unique violation"):
        main_module._run_adaptive_mutation_transaction(lambda _conn: None)
    assert attempts == 1


def test_adaptive_mutation_retry_checks_cancellation_before_second_attempt(
    monkeypatch,
) -> None:
    work_calls = 0
    cancelled = False

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()

    def work(_conn) -> None:
        nonlocal work_calls, cancelled
        work_calls += 1
        cancelled = True
        raise _PostgresFailure("40001", "serialization failure")

    monkeypatch.setattr(main_module, "get_conn", connection)
    with pytest.raises(main_module.AssessmentCancelledError):
        main_module._run_adaptive_mutation_transaction(
            work,
            should_cancel=lambda: cancelled,
        )
    assert work_calls == 1


@pytest.mark.parametrize(
    "semantic_error",
    [
        ValueError("invalid adaptive input"),
        main_module.DatabaseIntegrityError("integrity failure"),
        main_module.HTTPException(status_code=409, detail="conflict"),
    ],
)
def test_adaptive_mutation_retry_never_replays_semantic_errors(
    monkeypatch,
    semantic_error: Exception,
) -> None:
    work_calls = 0
    semantic_error.__cause__ = _PostgresFailure("40001", "serialization failure")

    @contextmanager
    def connection(*, transactional: bool = False):
        assert transactional is True
        yield object()

    def work(_conn) -> None:
        nonlocal work_calls
        work_calls += 1
        raise semantic_error

    monkeypatch.setattr(main_module, "get_conn", connection)
    with pytest.raises(semantic_error.__class__):
        main_module._run_adaptive_mutation_transaction(work)
    assert work_calls == 1


class TestAssessmentApi:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        get_settings.cache_clear()
        db_module._db_ready = False
        main_module.settings = get_settings()
        main_module._rate_limit_hits.clear()
        self.client = TestClient(main_module.app)
        self._seed()

    def teardown_method(self) -> None:
        self.client.close()
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        get_settings.cache_clear()
        db_module._db_ready = False
        main_module.settings = get_settings()
        self.temp_dir.cleanup()

    @property
    def headers_a(self) -> dict[str, str]:
        return {OWNER_HEADER: OWNER_A}

    def _seed(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO materials "
                "(id, subject_tag, raw_text, source_type, knowledge_level, created_at) "
                "VALUES (?, 'physics', 'physics', 'topic', 'beginner', ?)",
                (MATERIAL, "2026-07-09T00:00:00+00:00"),
            )
            for index in range(5):
                concept_id = f"api-concept-{index}"
                video_id = f"api-video-{index}"
                reel_id = f"api-reel-{index}"
                conn.execute(
                    "INSERT INTO concepts "
                    "(id, material_id, title, keywords_json, summary, created_at) "
                    "VALUES (?, ?, ?, '[]', '', ?)",
                    (concept_id, MATERIAL, f"Concept {index}", "2026-07-09T00:00:00+00:00"),
                )
                conn.execute(
                    "INSERT INTO videos (id, title, created_at) VALUES (?, ?, ?)",
                    (video_id, video_id, "2026-07-09T00:00:00+00:00"),
                )
                conn.execute(
                    "INSERT INTO reels "
                    "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                    "transcript_snippet, takeaways_json, ai_summary, match_reason, "
                    "informativeness, base_score, difficulty, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, 180, ?, '[]', '', '', 1.0, 1.0, 0.5, ?)",
                    (
                        reel_id,
                        MATERIAL,
                        concept_id,
                        video_id,
                        f"https://example.test/{video_id}",
                        "Gravity pulls an object toward Earth with force.",
                        f"2026-07-09T00:00:0{index}+00:00",
                    ),
                )
                assert store_reel_assessment_question(
                    conn,
                    reel_id=reel_id,
                    prompt="What pulls an object toward Earth?",
                    options=["Gravity", "Sound", "Heat", "Light"],
                    correct_index=0,
                    explanation="Gravity pulls an object toward Earth with force.",
                )

    def test_progress_next_pending_answer_contract_and_privacy(self) -> None:
        progress = self.client.post(
            "/api/reels/api-reel-0/progress",
            headers=self.headers_a,
            json={"max_fraction": 1.0},
        )
        assert progress.status_code == 200
        assert progress.json()["assessment_ready"] is False
        assert {
            "reel_id",
            "completed",
            "newly_completed",
            "assessment_ready",
            "information_units",
            "readiness_threshold",
        } == set(progress.json())

        scrolls = []
        for index in range(5):
            response = self.client.post(
                f"/api/reels/api-reel-{index}/scroll",
                headers=self.headers_a,
            )
            assert response.status_code == 200
            scrolls.append(response.json())
            if scrolls[-1]["assessment_ready"]:
                break
        assert scrolls[-1]["assessment_ready"] is True
        assert scrolls[-1]["scroll_count"] == scrolls[-1]["cadence_target"]
        assert 3 <= scrolls[-1]["cadence_target"] <= 5
        assert {
            "reel_id",
            "material_id",
            "newly_scrolled",
            "assessment_ready",
            "scroll_count",
            "cadence_target",
        } == set(scrolls[-1])
        duplicate = self.client.post(
            "/api/reels/api-reel-0/scroll", headers=self.headers_a
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["newly_scrolled"] is False
        assert duplicate.json()["scroll_count"] == scrolls[-1]["scroll_count"]

        created = self.client.post(
            "/api/assessments/next",
            headers=self.headers_a,
            json={"material_id": MATERIAL},
        )
        assert created.status_code == 200
        body = created.json()
        assert body["status"] == "ready"
        assert body["session"]["question_count"] == 3
        for question in body["session"]["questions"]:
            assert "correct_index" not in question
            assert "explanation" not in question

        pending = self.client.get(
            "/api/assessments/pending",
            headers=self.headers_a,
            params={"material_id": MATERIAL},
        )
        assert pending.status_code == 200
        assert pending.json()["session"]["id"] == body["session"]["id"]

        session = body["session"]
        result = None
        for question in session["questions"]:
            response = self.client.post(
                f"/api/assessments/{session['id']}/answer",
                headers=self.headers_a,
                json={"question_id": question["id"], "choice_index": 0},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["correct"] is True
            assert result["correct_index"] == 0
            assert result["explanation"]
        assert result is not None
        assert result["session"]["status"] == "completed"
        assert result["session"]["score"] == 1.0
        assert len(result["session"]["understood_concepts"]) == 3
        with db_module.get_conn() as conn:
            progress = db_module.fetch_one(
                conn,
                "SELECT global_adjustment FROM learner_material_progress "
                "WHERE material_id = ?",
                (MATERIAL,),
            )
        assert progress is not None
        assert float(progress["global_adjustment"]) == 0.2

    def test_feedback_retry_after_lost_commit_ack_does_not_double_revision(
        self,
    ) -> None:
        flaky_conn, attempts = _fail_first_commit_acknowledgement(
            main_module.get_conn
        )
        with (
            mock.patch.object(main_module, "get_conn", flaky_conn),
            mock.patch.object(
                main_module.reel_service,
                "update_level_adjustment",
                wraps=main_module.reel_service.update_level_adjustment,
            ) as update_mastery,
        ):
            response = self.client.post(
                "/api/reels/feedback",
                headers=self.headers_a,
                json={
                    "reel_id": "api-reel-0",
                    "helpful": True,
                    "confusing": False,
                    "saved": False,
                },
            )

        assert response.status_code == 200
        assert len(attempts) == 2
        assert update_mastery.call_count == 1
        with db_module.get_conn() as conn:
            progress = db_module.fetch_one(
                conn,
                "SELECT feedback_revision FROM learner_material_progress "
                "WHERE material_id = ?",
                (MATERIAL,),
            )
            feedback_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM reel_feedback "
                "WHERE reel_id = 'api-reel-0'",
            )
        assert progress is not None
        assert int(progress["feedback_revision"]) == 1
        assert int((feedback_count or {})["count"]) == 1

    def test_feedback_retry_preserves_signal_after_unrelated_revision_advance(
        self,
    ) -> None:
        learner_id = "owner:" + hashlib.sha256(OWNER_A.encode("utf-8")).hexdigest()
        real_get_conn = main_module.get_conn
        with real_get_conn(transactional=True) as conn:
            main_module.reel_service.learner_progress(conn, MATERIAL, learner_id)

        transaction_attempts = 0

        @contextmanager
        def connection(*, transactional: bool = False):
            nonlocal transaction_attempts
            if not transactional:
                with real_get_conn() as conn:
                    yield conn
                return
            transaction_attempts += 1
            try:
                with real_get_conn(transactional=True) as conn:
                    yield conn
                    if transaction_attempts == 1:
                        raise _PostgresFailure("40001", "serialization failure")
            except _PostgresFailure:
                if transaction_attempts == 1:
                    with real_get_conn(transactional=True) as concurrent_conn:
                        main_module.reel_service.record_feedback(
                            concurrent_conn,
                            reel_id="api-reel-1",
                            helpful=False,
                            confusing=True,
                            rating=None,
                            saved=False,
                            learner_id=learner_id,
                        )
                raise

        with mock.patch.object(main_module, "get_conn", connection):
            response = self.client.post(
                "/api/reels/feedback",
                headers=self.headers_a,
                json={
                    "reel_id": "api-reel-0",
                    "helpful": True,
                    "confusing": False,
                    "saved": False,
                },
            )

        assert response.status_code == 200
        assert transaction_attempts == 2
        with real_get_conn() as conn:
            progress = db_module.fetch_one(
                conn,
                "SELECT feedback_revision FROM learner_material_progress "
                "WHERE learner_id = ? AND material_id = ?",
                (learner_id, MATERIAL),
            )
            feedback = db_module.fetch_one(
                conn,
                "SELECT helpful, confusing FROM reel_feedback "
                "WHERE learner_id = ? AND reel_id = 'api-reel-0'",
                (learner_id,),
            )
        assert progress is not None
        assert int(progress["feedback_revision"]) == 2
        assert feedback is not None
        assert int(feedback["helpful"]) == 1
        assert int(feedback["confusing"]) == 0

    def test_progress_and_scroll_retries_keep_one_monotonic_row(self) -> None:
        flaky_progress_conn, progress_attempts = _fail_first_commit_acknowledgement(
            main_module.get_conn
        )
        with mock.patch.object(main_module, "get_conn", flaky_progress_conn):
            progress_response = self.client.post(
                "/api/reels/api-reel-0/progress",
                headers=self.headers_a,
                json={"max_fraction": 1.0},
            )
        assert progress_response.status_code == 200
        assert len(progress_attempts) == 2

        flaky_scroll_conn, scroll_attempts = _fail_first_commit_acknowledgement(
            main_module.get_conn
        )
        with mock.patch.object(main_module, "get_conn", flaky_scroll_conn):
            scroll_response = self.client.post(
                "/api/reels/api-reel-0/scroll",
                headers=self.headers_a,
            )
        assert scroll_response.status_code == 200
        assert len(scroll_attempts) == 2

        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT max_fraction, scrolled_at, completed_at "
                "FROM learner_reel_progress WHERE reel_id = 'api-reel-0'",
            )
        assert len(rows) == 1
        assert float(rows[0]["max_fraction"]) == 1.0
        assert rows[0]["completed_at"]
        assert rows[0]["scrolled_at"]

    def test_session_and_final_answer_retries_do_not_duplicate_quiz_state(
        self,
    ) -> None:
        ready = False
        for index in range(5):
            response = self.client.post(
                f"/api/reels/api-reel-{index}/scroll",
                headers=self.headers_a,
            )
            assert response.status_code == 200
            ready = bool(response.json()["assessment_ready"])
            if ready:
                break
        assert ready

        flaky_session_conn, session_attempts = _fail_first_commit_acknowledgement(
            main_module.get_conn
        )
        with mock.patch.object(main_module, "get_conn", flaky_session_conn):
            created_response = self.client.post(
                "/api/assessments/next",
                headers=self.headers_a,
                json={"material_id": MATERIAL},
            )
        assert created_response.status_code == 200
        assert len(session_attempts) == 2
        session = created_response.json()["session"]
        assert session is not None

        for question in session["questions"][:-1]:
            answer = self.client.post(
                f"/api/assessments/{session['id']}/answer",
                headers=self.headers_a,
                json={"question_id": question["id"], "choice_index": 0},
            )
            assert answer.status_code == 200

        final_question = session["questions"][-1]
        flaky_answer_conn, answer_attempts = _fail_first_commit_acknowledgement(
            main_module.get_conn
        )
        with mock.patch.object(main_module, "get_conn", flaky_answer_conn):
            final_response = self.client.post(
                f"/api/assessments/{session['id']}/answer",
                headers=self.headers_a,
                json={"question_id": final_question["id"], "choice_index": 0},
            )
        assert final_response.status_code == 200
        assert len(answer_attempts) == 2
        assert final_response.json()["session"]["status"] == "completed"

        with db_module.get_conn() as conn:
            session_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM assessment_sessions",
            )
            question_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM assessment_session_questions "
                "WHERE session_id = ?",
                (session["id"],),
            )
            attempt_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM assessment_attempts "
                "WHERE session_id = ?",
                (session["id"],),
            )
            outcome_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM assessment_concept_outcomes "
                "WHERE session_id = ?",
                (session["id"],),
            )
            progress = db_module.fetch_one(
                conn,
                "SELECT feedback_revision FROM learner_material_progress "
                "WHERE material_id = ?",
                (MATERIAL,),
            )
        expected_questions = int(session["question_count"])
        assert int((session_count or {})["count"]) == 1
        assert int((question_count or {})["count"]) == expected_questions
        assert int((attempt_count or {})["count"]) == expected_questions
        assert int((outcome_count or {})["count"]) == expected_questions
        assert int((progress or {})["feedback_revision"]) == 1

    def test_other_learner_cannot_resume_or_answer_session(self) -> None:
        ready = False
        for index in range(5):
            response = self.client.post(
                f"/api/reels/api-reel-{index}/scroll",
                headers=self.headers_a,
            )
            assert response.status_code == 200
            ready = bool(response.json()["assessment_ready"])
            if ready:
                break
        assert ready is True
        created = self.client.post(
            "/api/assessments/next",
            headers=self.headers_a,
            json={"material_id": MATERIAL},
        ).json()
        other_headers = {OWNER_HEADER: OWNER_B}
        pending = self.client.get(
            "/api/assessments/pending",
            headers=other_headers,
            params={"material_id": MATERIAL},
        )
        assert pending.status_code == 200
        assert pending.json()["session"] is None
        forbidden = self.client.post(
            f"/api/assessments/{created['session']['id']}/answer",
            headers=other_headers,
            json={
                "question_id": created["session"]["questions"][0]["id"],
                "choice_index": 0,
            },
        )
        assert forbidden.status_code == 404

    def test_final_answer_rolls_back_all_quiz_adaptation_on_later_failure(
        self,
    ) -> None:
        ready = False
        for index in range(5):
            response = self.client.post(
                f"/api/reels/api-reel-{index}/scroll",
                headers=self.headers_a,
            )
            assert response.status_code == 200
            ready = bool(response.json()["assessment_ready"])
            if ready:
                break
        assert ready
        session = self.client.post(
            "/api/assessments/next",
            headers=self.headers_a,
            json={"material_id": MATERIAL},
        ).json()["session"]
        for question in session["questions"][:2]:
            response = self.client.post(
                f"/api/assessments/{session['id']}/answer",
                headers=self.headers_a,
                json={"question_id": question["id"], "choice_index": 0},
            )
            assert response.status_code == 200

        final_question = session["questions"][2]
        with (
            mock.patch.object(
                main_module.reel_service,
                "update_level_adjustment",
                side_effect=RuntimeError("post-assessment failure"),
            ),
            pytest.raises(RuntimeError, match="post-assessment failure"),
        ):
            self.client.post(
                f"/api/assessments/{session['id']}/answer",
                headers=self.headers_a,
                json={"question_id": final_question["id"], "choice_index": 0},
            )

        with db_module.get_conn() as conn:
            stored = db_module.fetch_one(
                conn,
                "SELECT status, current_index FROM assessment_sessions WHERE id = ?",
                (session["id"],),
            )
            attempts = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM assessment_attempts WHERE session_id = ?",
                (session["id"],),
            )
            outcomes = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM assessment_concept_outcomes "
                "WHERE session_id = ?",
                (session["id"],),
            )
        assert stored == {"status": "pending", "current_index": 2}
        assert attempts == {"count": 2}
        assert outcomes == {"count": 0}

    def test_adaptation_lock_serializes_real_sqlite_connections(self) -> None:
        learner_id = "owner:" + hashlib.sha256(OWNER_A.encode("utf-8")).hexdigest()
        first_locked = threading.Event()
        release_first = threading.Event()
        second_locked = threading.Event()
        errors: list[BaseException] = []

        def hold_first_lock() -> None:
            try:
                with db_module.get_conn(transactional=True) as conn:
                    main_module._lock_learner_adaptation(
                        conn,
                        material_id=MATERIAL,
                        learner_id=learner_id,
                    )
                    conn.execute(
                        "UPDATE learner_material_progress "
                        "SET selected_level = 'advanced', global_adjustment = 0.2, "
                        "feedback_revision = 7 "
                        "WHERE learner_id = ? AND material_id = ?",
                        (learner_id, MATERIAL),
                    )
                    first_locked.set()
                    assert release_first.wait(3)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        def wait_for_same_lock() -> None:
            try:
                assert first_locked.wait(3)
                with db_module.get_conn(transactional=True) as conn:
                    main_module._lock_learner_adaptation(
                        conn,
                        material_id=MATERIAL,
                        learner_id=learner_id,
                    )
                    second_locked.set()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        first = threading.Thread(target=hold_first_lock)
        second = threading.Thread(target=wait_for_same_lock)
        first.start()
        assert first_locked.wait(3)
        second.start()
        assert not second_locked.wait(0.2)
        release_first.set()
        first.join(timeout=3)
        second.join(timeout=3)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert second_locked.is_set()
        with db_module.get_conn() as conn:
            progress = db_module.fetch_one(
                conn,
                "SELECT selected_level, global_adjustment, feedback_revision "
                "FROM learner_material_progress "
                "WHERE learner_id = ? AND material_id = ?",
                (learner_id, MATERIAL),
            )
        assert progress == {
            "selected_level": "advanced",
            "global_adjustment": 0.2,
            "feedback_revision": 7,
        }

    def test_progress_requires_a_real_study_reel(self) -> None:
        response = self.client.post(
            "/api/reels/community-only/progress",
            headers=self.headers_a,
            json={"max_fraction": 1.0},
        )
        assert response.status_code == 404
        scroll = self.client.post(
            "/api/reels/community-only/scroll",
            headers=self.headers_a,
        )
        assert scroll.status_code == 404
