"""Focused adaptive recall-check service tests (SQLite, no network)."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

import backend.app.services.assessments as assessments_module
from backend.app.db import SCHEMA
from backend.app.services.assessments import (
    AssessmentCancelledError,
    AssessmentService,
    _question_fingerprint,
    store_reel_assessment_question,
)


MATERIAL = "material-recall"
LEARNER = "owner:learner-a"


@pytest.fixture()
def conn():
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript(SCHEMA)
    database.execute(
        "INSERT INTO materials "
        "(id, subject_tag, raw_text, source_type, knowledge_level, created_at) "
        "VALUES (?, 'physics', 'physics', 'topic', 'beginner', '2026-07-09T00:00:00+00:00')",
        (MATERIAL,),
    )
    yield database
    database.close()


def _seed_reel(
    conn,
    *,
    reel_id: str,
    concept_id: str,
    video_id: str,
    duration: float = 180.0,
    informativeness: float = 1.0,
    difficulty: float = 0.5,
    with_question: bool = True,
) -> None:
    if not conn.execute("SELECT 1 FROM concepts WHERE id = ?", (concept_id,)).fetchone():
        conn.execute(
            "INSERT INTO concepts "
            "(id, material_id, title, keywords_json, summary, created_at) "
            "VALUES (?, ?, ?, '[]', '', '2026-07-09T00:00:00+00:00')",
            (concept_id, MATERIAL, f"Concept {concept_id}"),
        )
    if not conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone():
        conn.execute(
            "INSERT INTO videos (id, title, created_at) VALUES (?, ?, '2026-07-09T00:00:00+00:00')",
            (video_id, video_id),
        )
    transcript = f"Gravity in {concept_id} pulls an object toward Earth with a measurable force."
    conn.execute(
        "INSERT INTO reels "
        "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
        "transcript_snippet, takeaways_json, ai_summary, match_reason, informativeness, "
        "base_score, difficulty, created_at) "
        "VALUES (?, ?, ?, ?, ?, 0, ?, ?, '[]', '', '', ?, 1.0, ?, ?)",
        (
            reel_id,
            MATERIAL,
            concept_id,
            video_id,
            f"https://example.test/{video_id}",
            duration,
            transcript,
            informativeness,
            difficulty,
            f"2026-07-09T00:00:{len(reel_id):02d}+00:00",
        ),
    )
    if with_question:
        stored = store_reel_assessment_question(
            conn,
            reel_id=reel_id,
            prompt="What pulls an object toward Earth?",
            options=["Gravity", "Sound", "Heat", "Light"],
            correct_index=0,
            explanation="Gravity pulls an object toward Earth with force.",
        )
        assert stored is not None


def _complete(service: AssessmentService, conn, reel_id: str, learner: str = LEARNER):
    return service.record_progress(
        conn, learner_id=learner, reel_id=reel_id, max_fraction=1.0
    )


def test_progress_is_idempotent_and_rejects_non_study_reels(conn) -> None:
    _seed_reel(conn, reel_id="r1", concept_id="c1", video_id="v1")
    service = AssessmentService()
    first = service.record_progress(
        conn, learner_id=LEARNER, reel_id="r1", max_fraction=0.79
    )
    second = service.record_progress(
        conn, learner_id=LEARNER, reel_id="r1", max_fraction=0.80
    )
    third = service.record_progress(
        conn, learner_id=LEARNER, reel_id="r1", max_fraction=0.20
    )
    assert first["completed"] is False
    assert second["completed"] is True and second["newly_completed"] is True
    assert third["completed"] is True and third["newly_completed"] is False
    row = conn.execute(
        "SELECT max_fraction, completed_at FROM learner_reel_progress "
        "WHERE learner_id = ? AND reel_id = 'r1'",
        (LEARNER,),
    ).fetchone()
    assert float(row["max_fraction"]) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="reel_id not found"):
        service.record_progress(
            conn, learner_id=LEARNER, reel_id="community-only", max_fraction=1.0
        )


def test_short_clips_do_not_create_a_fixed_two_clip_cadence(conn) -> None:
    service = AssessmentService(question_generator=lambda *_args: None)
    for index in range(2):
        reel_id = f"short-{index}"
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id=f"short-c{index}",
            video_id=f"short-v{index}",
            duration=20.0,
            informativeness=0.6,
        )
        result = _complete(service, conn, reel_id)
    assert result["information_units"] < result["readiness_threshold"]
    assert result["assessment_ready"] is False


def test_dense_clips_create_and_resume_private_question_session(conn) -> None:
    service = AssessmentService()
    for index in range(2):
        reel_id = f"dense-{index}"
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id=f"dense-c{index}",
            video_id=f"dense-v{index}",
        )
        progress = _complete(service, conn, reel_id)
    assert progress["assessment_ready"] is True

    created = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    resumed = service.pending(conn, learner_id=LEARNER, material_id=MATERIAL)
    assert created["status"] == "ready"
    assert resumed["status"] == "pending"
    assert resumed["session"]["id"] == created["session"]["id"]
    assert 1 <= created["session"]["question_count"] <= 4
    for question in created["session"]["questions"]:
        assert len(question["options"]) == 4
        assert "correct_index" not in question
        assert "explanation" not in question


def test_question_storage_is_immutable_for_a_reel_fingerprint(conn) -> None:
    _seed_reel(conn, reel_id="immutable", concept_id="immutable-c", video_id="immutable-v")
    original = conn.execute(
        "SELECT prompt, options_json, correct_index, explanation "
        "FROM reel_assessment_questions WHERE reel_id = 'immutable'"
    ).fetchone()
    replay = store_reel_assessment_question(
        conn,
        reel_id="immutable",
        prompt="Which force pulls an object toward Earth?",
        options=["Gravity", "Wind", "Heat", "Light"],
        correct_index=1,
        explanation="Gravity pulls an object toward Earth with force.",
    )
    stored = conn.execute(
        "SELECT prompt, options_json, correct_index, explanation "
        "FROM reel_assessment_questions WHERE reel_id = 'immutable'"
    ).fetchone()
    assert replay is not None
    assert tuple(stored) == tuple(original)
    assert replay["prompt"] == original["prompt"]
    assert store_reel_assessment_question(
        conn,
        reel_id="immutable",
        prompt="Which force pulls an object toward Earth?",
        options=["Gravity", "Wind", "Heat", "Light"],
        correct_index=0.5,
        explanation="Gravity pulls an object toward Earth with force.",
        fingerprint="non-integral-answer",
    ) is None


def test_answer_reveals_key_then_applies_concept_outcomes_once(conn) -> None:
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"answer-{index}",
            concept_id=f"answer-c{index}",
            video_id=f"answer-v{index}",
            difficulty=0.8,
        )
        _complete(service, conn, f"answer-{index}")
    wrapper = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    session = wrapper["session"]
    first, second = session["questions"][:2]
    first_result = service.answer(
        conn,
        learner_id=LEARNER,
        session_id=session["id"],
        question_id=first["id"],
        choice_index=0,
    )
    assert first_result["correct"] is True
    assert first_result["correct_index"] == 0
    assert first_result["explanation"]
    final = service.answer(
        conn,
        learner_id=LEARNER,
        session_id=session["id"],
        question_id=second["id"],
        choice_index=1,
    )
    assert final["session"]["status"] == "completed"
    assert final["session"]["score"] == pytest.approx(0.5)
    assert len(final["session"]["understood_concepts"]) == 1
    assert len(final["session"]["revisit_concepts"]) == 1
    outcomes = conn.execute(
        "SELECT adjustment FROM assessment_concept_outcomes "
        "WHERE session_id = ? ORDER BY adjustment",
        (session["id"],),
    ).fetchall()
    assert [float(row["adjustment"]) for row in outcomes] == [-0.12, 0.08]

    duplicate = service.answer(
        conn,
        learner_id=LEARNER,
        session_id=session["id"],
        question_id=second["id"],
        choice_index=0,
    )
    assert duplicate["correct"] is False
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_concept_outcomes WHERE session_id = ?",
        (session["id"],),
    ).fetchone()[0] == 2
    with pytest.raises(ValueError, match="choice_index"):
        service.answer(
            conn,
            learner_id=LEARNER,
            session_id=session["id"],
            question_id=first["id"],
            choice_index=4,
        )


def test_new_answers_must_follow_question_order(conn) -> None:
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"ordered-{index}",
            concept_id=f"ordered-c{index}",
            video_id=f"ordered-v{index}",
        )
        _complete(service, conn, f"ordered-{index}")
    session = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)["session"]
    first, second = session["questions"]
    with pytest.raises(ValueError, match="answered in order"):
        service.answer(
            conn,
            learner_id=LEARNER,
            session_id=session["id"],
            question_id=second["id"],
            choice_index=0,
        )
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_attempts WHERE session_id = ?",
        (session["id"],),
    ).fetchone()[0] == 0
    result = service.answer(
        conn,
        learner_id=LEARNER,
        session_id=session["id"],
        question_id=first["id"],
        choice_index=0,
    )
    assert result["session"]["answered_count"] == 1


def test_attempt_insert_race_reloads_the_winning_choice(conn, monkeypatch) -> None:
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"race-{index}",
            concept_id=f"race-c{index}",
            video_id=f"race-v{index}",
        )
        _complete(service, conn, f"race-{index}")
    session = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)["session"]
    question = session["questions"][0]
    original_execute = assessments_module.execute_modify
    injected = False

    def race_execute(connection, query, params=()):
        nonlocal injected
        if "INSERT INTO assessment_attempts" in query and not injected:
            injected = True
            winning_params = list(params)
            winning_params[4] = 1
            winning_params[5] = 0
            assert original_execute(connection, query, winning_params) == 1
            return 0
        return original_execute(connection, query, params)

    monkeypatch.setattr(assessments_module, "execute_modify", race_execute)
    result = service.answer(
        conn,
        learner_id=LEARNER,
        session_id=session["id"],
        question_id=question["id"],
        choice_index=0,
    )
    assert result["correct"] is False
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_attempts WHERE session_id = ? AND question_id = ?",
        (session["id"], question["id"]),
    ).fetchone()[0] == 1


def test_sessions_and_attempts_are_learner_isolated(conn) -> None:
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"isolated-{index}",
            concept_id=f"isolated-c{index}",
            video_id=f"isolated-v{index}",
        )
        _complete(service, conn, f"isolated-{index}", LEARNER)
        _complete(service, conn, f"isolated-{index}", "owner:learner-b")
    first = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    second = service.next_session(
        conn, learner_id="owner:learner-b", material_id=MATERIAL
    )
    assert first["session"]["id"] != second["session"]["id"]
    with pytest.raises(ValueError, match="not found"):
        service.answer(
            conn,
            learner_id="owner:learner-b",
            session_id=first["session"]["id"],
            question_id=first["session"]["questions"][0]["id"],
            choice_index=0,
        )


def test_snooze_requires_new_information_without_penalty(conn) -> None:
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"snooze-{index}",
            concept_id=f"snooze-c{index}",
            video_id=f"snooze-v{index}",
        )
        _complete(service, conn, f"snooze-{index}")
    session = service.next_session(
        conn, learner_id=LEARNER, material_id=MATERIAL
    )["session"]
    assert service.snooze(
        conn, learner_id=LEARNER, session_id=session["id"]
    ) == {"status": "snoozed", "assessment_ready": False}
    assert service.pending(
        conn, learner_id=LEARNER, material_id=MATERIAL
    )["assessment_ready"] is False
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_concept_outcomes WHERE learner_id = ?",
        (LEARNER,),
    ).fetchone()[0] == 0

    _seed_reel(
        conn,
        reel_id="snooze-new",
        concept_id="snooze-new-c",
        video_id="snooze-new-v",
    )
    progress = _complete(service, conn, "snooze-new")
    assert progress["assessment_ready"] is True


def test_snooze_reports_an_already_completed_session_as_completed(conn) -> None:
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"completed-snooze-{index}",
            concept_id=f"completed-snooze-c{index}",
            video_id=f"completed-snooze-v{index}",
        )
        _complete(service, conn, f"completed-snooze-{index}")

    session = service.next_session(
        conn, learner_id=LEARNER, material_id=MATERIAL
    )["session"]
    for question in session["questions"]:
        service.answer(
            conn,
            learner_id=LEARNER,
            session_id=session["id"],
            question_id=question["id"],
            choice_index=0,
        )

    assert service.snooze(
        conn, learner_id=LEARNER, session_id=session["id"]
    ) == {"status": "completed", "assessment_ready": False}


def test_legacy_backfill_is_one_batch_cached_and_malformed_items_are_isolated(conn) -> None:
    calls: list[list[str]] = []

    def generator(rows, _should_cancel=None):
        calls.append([str(row["reel_id"]) for row in rows])
        return {
            "questions": [
                {
                    "reel_id": rows[0]["reel_id"],
                    "prompt": "What pulls an object toward Earth?",
                    "options": ["Gravity", "Sound", "Heat", "Light"],
                    "correct_index": 0,
                    "explanation": "Gravity pulls an object toward Earth.",
                },
                {
                    "reel_id": rows[1]["reel_id"],
                    "prompt": "Malformed",
                    "options": ["Gravity", "Gravity", "Heat", "Light"],
                    "correct_index": 0,
                    "explanation": "Gravity pulls an object toward Earth.",
                },
            ]
        }

    service = AssessmentService(question_generator=generator)
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"legacy-{index}",
            concept_id=f"legacy-c{index}",
            video_id=f"legacy-v{index}",
            with_question=False,
        )
        _complete(service, conn, f"legacy-{index}")
    assert len(calls) == 1
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM llm_cache WHERE cache_key LIKE 'assessment_question_backfill:%'"
    ).fetchone()[0] == 2

    conn.execute("DELETE FROM reel_assessment_questions")
    service._ensure_question_pool(
        conn,
        learner_id=LEARNER,
        material_id=MATERIAL,
        completion_rows=service._completed_rows(conn, LEARNER, MATERIAL),
        should_cancel=None,
    )
    assert len(calls) == 1
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 1


def test_legacy_backfill_scans_past_cached_old_rows(conn) -> None:
    calls: list[list[str]] = []

    def generator(rows, _should_cancel=None):
        calls.append([str(row["reel_id"]) for row in rows])
        return {
            "questions": [
                {
                    "reel_id": row["reel_id"],
                    "prompt": "What pulls an object toward Earth?",
                    "options": ["Gravity", "Sound", "Heat", "Light"],
                    "correct_index": 0,
                    "explanation": "Gravity pulls an object toward Earth.",
                }
                for row in rows
            ]
        }

    service = AssessmentService(question_generator=generator)
    for index in range(14):
        reel_id = f"starved-{index:02d}"
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id=f"starved-c{index:02d}",
            video_id=f"starved-v{index:02d}",
            with_question=False,
        )
        timestamp = f"2026-07-09T00:01:{index:02d}+00:00"
        conn.execute(
            "INSERT INTO learner_reel_progress "
            "(learner_id, reel_id, material_id, max_fraction, completed_at, created_at, updated_at) "
            "VALUES (?, ?, ?, 1.0, ?, ?, ?)",
            (LEARNER, reel_id, MATERIAL, timestamp, timestamp, timestamp),
        )
    rows = service._completed_rows(conn, LEARNER, MATERIAL)
    for row in rows[:12]:
        service._write_cached_backfill(conn, _question_fingerprint(row), None)
    service._ensure_question_pool(
        conn,
        learner_id=LEARNER,
        material_id=MATERIAL,
        completion_rows=rows,
        should_cancel=None,
    )
    assert calls == [["starved-13", "starved-12"]]
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 2


def test_latest_check_accuracy_controls_the_question_count_bonus(conn) -> None:
    service = AssessmentService()
    state = {
        "information_units": 3.0,
        "completion_rows": [{"concept_id": "c1"}, {"concept_id": "c2"}],
        "recent_accuracy": 0.50,
        "rolling_accuracy": 0.95,
        "recent_session_accuracies": [0.50, 1.0],
        "available_questions": [{"id": str(index)} for index in range(4)],
    }
    assert service._desired_question_count(conn, state) == 3
    state["recent_accuracy"] = 0.95
    state["rolling_accuracy"] = 0.50
    assert service._desired_question_count(conn, state) == 2


def test_session_creation_cancels_before_writes_and_rolls_back_child_failure(
    conn, monkeypatch
) -> None:
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"atomic-{index}",
            concept_id=f"atomic-c{index}",
            video_id=f"atomic-v{index}",
        )
        _complete(service, conn, f"atomic-{index}")
    with pytest.raises(AssessmentCancelledError):
        service.next_session(
            conn,
            learner_id=LEARNER,
            material_id=MATERIAL,
            should_cancel=lambda: True,
        )
    assert conn.execute("SELECT COUNT(*) FROM assessment_sessions").fetchone()[0] == 0

    original_execute = assessments_module.execute_modify
    child_writes = 0

    def failing_execute(connection, query, params=()):
        nonlocal child_writes
        if "INSERT INTO assessment_session_questions" in query:
            child_writes += 1
            if child_writes == 2:
                raise RuntimeError("injected child failure")
        return original_execute(connection, query, params)

    monkeypatch.setattr(assessments_module, "execute_modify", failing_execute)
    with pytest.raises(RuntimeError, match="injected child failure"):
        service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    assert conn.execute("SELECT COUNT(*) FROM assessment_sessions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM assessment_session_questions").fetchone()[0] == 0


def test_default_backfill_forwards_the_cancellation_probe(monkeypatch) -> None:
    probe = lambda: False
    observed: dict[str, object] = {}

    def fake_llm_json(*_args, **kwargs):
        observed["probe"] = kwargs.get("should_cancel")
        return SimpleNamespace(model_dump=lambda: {"questions": []})

    monkeypatch.setattr(
        "backend.app.clip_engine.clipper.llm.llm_json", fake_llm_json
    )
    result = AssessmentService()._default_question_generator(
        [{"reel_id": "r1", "concept_title": "Gravity", "transcript_snippet": "Gravity."}],
        probe,
    )
    assert result == {"questions": []}
    assert observed["probe"] is probe


def test_backfill_observes_cancellation_before_any_cache_or_question_write(conn) -> None:
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"cancel-{index}",
            concept_id=f"cancel-c{index}",
            video_id=f"cancel-v{index}",
            with_question=False,
        )
    cancelled = False

    def generator(_rows, _probe=None):
        nonlocal cancelled
        cancelled = True
        return {"questions": []}

    service = AssessmentService(question_generator=generator)
    service.record_progress(
        conn, learner_id=LEARNER, reel_id="cancel-0", max_fraction=1.0
    )
    with pytest.raises(AssessmentCancelledError):
        service.record_progress(
            conn,
            learner_id=LEARNER,
            reel_id="cancel-1",
            max_fraction=1.0,
            should_cancel=lambda: cancelled,
        )
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM llm_cache WHERE cache_key LIKE 'assessment_question_backfill:%'"
    ).fetchone()[0] == 0
