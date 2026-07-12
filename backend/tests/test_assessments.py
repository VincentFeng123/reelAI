"""Focused adaptive recall-check service tests (SQLite, no network)."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

import backend.app.services.assessments as assessments_module
from backend.app.db import SCHEMA
from backend.app.clip_engine.errors import ProviderTransientError
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
    progress = service.record_progress(
        conn, learner_id=learner, reel_id=reel_id, max_fraction=1.0
    )
    scroll = service.record_scroll(conn, learner_id=learner, reel_id=reel_id)
    return {**progress, **scroll}


def _seed_completed_accuracy(
    conn,
    *,
    session_id: str,
    correct_count: int,
    question_count: int = 10,
    learner: str = LEARNER,
    completed_at: str = "2026-07-08T00:00:00+00:00",
) -> None:
    conn.execute(
        "INSERT INTO assessment_sessions "
        "(id, learner_id, material_id, status, current_index, question_count, "
        "correct_count, information_units, readiness_threshold, created_at, "
        "updated_at, completed_at) "
        "VALUES (?, ?, ?, 'completed', ?, ?, ?, 0, 3, ?, ?, ?)",
        (
            session_id,
            learner,
            MATERIAL,
            question_count,
            question_count,
            correct_count,
            completed_at,
            completed_at,
            completed_at,
        ),
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
        "SELECT max_fraction, scrolled_at, completed_at FROM learner_reel_progress "
        "WHERE learner_id = ? AND reel_id = 'r1'",
        (LEARNER,),
    ).fetchone()
    assert float(row["max_fraction"]) == pytest.approx(0.8)
    assert row["scrolled_at"] is None
    assert service.pending(
        conn, learner_id=LEARNER, material_id=MATERIAL
    )["assessment_ready"] is False
    with pytest.raises(ValueError, match="reel_id not found"):
        service.record_progress(
            conn, learner_id=LEARNER, reel_id="community-only", max_fraction=1.0
        )


def test_progress_preserves_scroll_written_after_its_initial_read(conn, monkeypatch) -> None:
    _seed_reel(conn, reel_id="progress-scroll-race", concept_id="race-c", video_id="race-v")
    service = AssessmentService()
    service.record_progress(
        conn,
        learner_id=LEARNER,
        reel_id="progress-scroll-race",
        max_fraction=0.25,
    )
    original_upsert = assessments_module.upsert
    injected_scroll_at = "2026-07-09T12:00:00+00:00"

    def interleaved_upsert(target_conn, table, data, pk="id"):
        if table == "learner_reel_progress":
            target_conn.execute(
                "UPDATE learner_reel_progress SET scrolled_at = ? "
                "WHERE learner_id = ? AND reel_id = ?",
                (injected_scroll_at, LEARNER, "progress-scroll-race"),
            )
        return original_upsert(target_conn, table, data, pk=pk)

    monkeypatch.setattr(assessments_module, "upsert", interleaved_upsert)
    service.record_progress(
        conn,
        learner_id=LEARNER,
        reel_id="progress-scroll-race",
        max_fraction=0.8,
    )

    row = conn.execute(
        "SELECT scrolled_at FROM learner_reel_progress "
        "WHERE learner_id = ? AND reel_id = ?",
        (LEARNER, "progress-scroll-race"),
    ).fetchone()
    assert row["scrolled_at"] == injected_scroll_at


def test_completion_does_not_drive_quiz_readiness(conn) -> None:
    service = AssessmentService()
    for index in range(5):
        reel_id = f"completed-only-{index}"
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id=f"completed-only-c{index}",
            video_id=f"completed-only-v{index}",
        )
        result = service.record_progress(
            conn, learner_id=LEARNER, reel_id=reel_id, max_fraction=1.0
        )
        assert result["completed"] is True
        assert result["assessment_ready"] is False


def test_scroll_is_idempotent_and_preserves_partial_watch_analytics(conn) -> None:
    _seed_reel(conn, reel_id="idempotent-scroll", concept_id="scroll-c", video_id="scroll-v")
    service = AssessmentService()
    service.record_progress(
        conn, learner_id=LEARNER, reel_id="idempotent-scroll", max_fraction=0.25
    )
    first = service.record_scroll(
        conn, learner_id=LEARNER, reel_id="idempotent-scroll"
    )
    second = service.record_scroll(
        conn, learner_id=LEARNER, reel_id="idempotent-scroll"
    )
    assert first["newly_scrolled"] is True
    assert second["newly_scrolled"] is False
    assert second["scroll_count"] == 1
    row = conn.execute(
        "SELECT max_fraction, scrolled_at, completed_at FROM learner_reel_progress "
        "WHERE learner_id = ? AND reel_id = 'idempotent-scroll'",
        (LEARNER,),
    ).fetchone()
    assert float(row["max_fraction"]) == pytest.approx(0.25)
    assert row["scrolled_at"]
    assert row["completed_at"] is None


def test_scroll_cadence_is_deterministic_adaptive_and_bounded(conn) -> None:
    service = AssessmentService()
    changed = [{"concept_id": "a"}, {"concept_id": "b"}]
    same = [{"concept_id": "a"}, {"concept_id": "a"}]
    common = {
        "learner_id": LEARNER,
        "material_id": MATERIAL,
        "window_cutoff": "",
    }
    neutral = service._cadence_target(
        **common, recent_accuracy=None, scroll_rows=changed
    )
    assert neutral == service._cadence_target(
        **common, recent_accuracy=None, scroll_rows=changed
    )
    low = service._cadence_target(
        **common, recent_accuracy=0.4, scroll_rows=changed
    )
    strong = service._cadence_target(
        **common, recent_accuracy=1.0, scroll_rows=changed
    )
    continuous = service._cadence_target(
        **common, recent_accuracy=None, scroll_rows=same
    )
    assert neutral == service._cadence_target(
        **common,
        recent_accuracy=None,
        scroll_rows=[*changed, {"concept_id": "b"}],
    )
    assert 2 <= low <= neutral <= strong <= 5
    assert neutral <= continuous <= 5


def test_distinct_scrolls_become_ready_at_backend_owned_target(conn) -> None:
    service = AssessmentService()
    results = []
    for index in range(5):
        reel_id = f"scroll-{index}"
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id="continuous-concept",
            video_id=f"scroll-v{index}",
        )
        results.append(service.record_scroll(conn, learner_id=LEARNER, reel_id=reel_id))
        if results[-1]["assessment_ready"]:
            break
    final = results[-1]
    assert final["assessment_ready"] is True
    assert final["scroll_count"] == final["cadence_target"]
    assert 2 <= final["cadence_target"] <= 5
    rows = conn.execute(
        "SELECT max_fraction, completed_at FROM learner_reel_progress "
        "WHERE learner_id = ? AND scrolled_at IS NOT NULL",
        (LEARNER,),
    ).fetchall()
    assert rows and all(float(row["max_fraction"]) == 0.0 for row in rows)
    assert all(row["completed_at"] is None for row in rows)


def test_dense_clips_create_and_resume_private_question_session(conn) -> None:
    service = AssessmentService()
    for index in range(3):
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
    assert created["session"]["question_count"] == 2
    for question in created["session"]["questions"]:
        assert len(question["options"]) == 4
        assert "correct_index" not in question
        assert "explanation" not in question


def test_session_uses_two_distinct_watched_reels_and_concepts(conn) -> None:
    service = AssessmentService()
    reel_ids = ["adaptive-0", "adaptive-1", "adaptive-2", "adaptive-unwatched"]
    for index, reel_id in enumerate(reel_ids):
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id=f"adaptive-c{index}",
            video_id=f"adaptive-v{index}",
        )

    for reel_id in reel_ids[:3]:
        _complete(service, conn, reel_id, LEARNER)
    session = service.next_session(
        conn, learner_id=LEARNER, material_id=MATERIAL
    )["session"]
    assert session["question_count"] == 2
    assert len({question["reel_id"] for question in session["questions"]}) == 2
    assert len({question["concept_id"] for question in session["questions"]}) == 2
    assert {question["reel_id"] for question in session["questions"]} <= set(reel_ids[:3])


def test_next_session_backfill_fills_a_second_recent_concept(conn) -> None:
    generated_reels: list[str] = []

    def generator(rows, _should_cancel=None):
        generated_reels.extend(str(row["reel_id"]) for row in rows)
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
    specs = [
        ("same-concept-0", "shared-concept", True),
        ("same-concept-1", "shared-concept", True),
        ("second-concept", "distinct-concept", False),
        ("same-concept-2", "shared-concept", True),
    ]
    for index, (reel_id, concept_id, with_question) in enumerate(specs):
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id=concept_id,
            video_id=f"concept-v{index}",
            with_question=with_question,
        )
        _complete(service, conn, reel_id)

    assert generated_reels == []
    session = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)[
        "session"
    ]
    assert generated_reels == ["second-concept"]
    assert {question["concept_id"] for question in session["questions"]} == {
        "shared-concept",
        "distinct-concept",
    }


def test_due_session_uses_one_question_when_only_one_is_grounded(conn) -> None:
    calls = 0

    def unavailable_generator(*_args):
        nonlocal calls
        calls += 1
        return None

    service = AssessmentService(question_generator=unavailable_generator)
    for index in range(3):
        _seed_reel(
            conn,
            reel_id=f"fallback-{index}",
            concept_id=f"fallback-c{index}",
            video_id=f"fallback-v{index}",
            with_question=index == 0,
        )
        progress = _complete(service, conn, f"fallback-{index}")

    assert progress["assessment_ready"] is True
    assert calls == 0
    created = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    assert created["status"] == "ready"
    assert created["session"]["question_count"] == 1
    assert created["session"]["questions"][0]["reel_id"] == "fallback-0"
    assert calls == 1


def test_transient_generation_failure_keeps_due_state_and_retries(conn) -> None:
    calls = 0

    def transient_generator(rows, _should_cancel=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderTransientError(
                "temporary assessment provider failure",
                provider="gemini",
                operation="assessment",
            )
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

    service = AssessmentService(question_generator=transient_generator)
    for index in range(3):
        _seed_reel(
            conn,
            reel_id=f"retry-{index}",
            concept_id=f"retry-c{index}",
            video_id=f"retry-v{index}",
            with_question=False,
        )
        progress = _complete(service, conn, f"retry-{index}")

    assert progress["assessment_ready"] is True
    assert calls == 0
    first = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    assert first["status"] == "not_ready"
    assert first["assessment_ready"] is True
    created = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    assert calls == 2
    assert created["status"] == "ready"
    assert created["session"]["question_count"] == 2


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


def test_question_storage_rejects_content_not_supported_by_the_watched_reel(conn) -> None:
    _seed_reel(
        conn,
        reel_id="grounded",
        concept_id="grounded-c",
        video_id="grounded-v",
        with_question=False,
    )
    assert store_reel_assessment_question(
        conn,
        reel_id="grounded",
        prompt="Which pigment captures sunlight during photosynthesis?",
        options=["Chlorophyll", "Hemoglobin", "Keratin", "Collagen"],
        correct_index=0,
        explanation="Chlorophyll captures sunlight for photosynthesis.",
    ) is None


def test_answer_reveals_key_then_applies_concept_outcomes_once(conn) -> None:
    _seed_completed_accuracy(conn, session_id="answer-history", correct_count=5)
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
    reset_state = service._readiness_state(conn, LEARNER, MATERIAL)
    assert reset_state["scroll_count"] == 0
    assert reset_state["assessment_ready"] is False
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
    _seed_completed_accuracy(conn, session_id="ordered-history", correct_count=5)
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
    _seed_completed_accuracy(conn, session_id="race-history", correct_count=5)
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
    _seed_completed_accuracy(
        conn, session_id="isolated-history-a", correct_count=5
    )
    _seed_completed_accuracy(
        conn,
        session_id="isolated-history-b",
        correct_count=5,
        learner="owner:learner-b",
    )
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
    _seed_completed_accuracy(conn, session_id="snooze-history", correct_count=5)
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

    progress = None
    for index in range(5):
        reel_id = f"snooze-new-{index}"
        _seed_reel(
            conn,
            reel_id=reel_id,
            concept_id=f"snooze-new-c{index}",
            video_id=f"snooze-new-v{index}",
        )
        progress = _complete(service, conn, reel_id)
        if progress["assessment_ready"]:
            break
    assert progress is not None
    assert progress["assessment_ready"] is True


def test_snooze_reports_an_already_completed_session_as_completed(conn) -> None:
    _seed_completed_accuracy(
        conn, session_id="completed-snooze-history", correct_count=5
    )
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


def test_malformed_backfill_cache_is_isolated_and_retried_at_next_session(conn) -> None:
    _seed_completed_accuracy(conn, session_id="backfill-history", correct_count=5)
    calls: list[list[str]] = []

    def generator(rows, _should_cancel=None):
        calls.append([str(row["reel_id"]) for row in rows])
        if len(calls) > 1:
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
    assert calls == []
    first = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)
    assert first["status"] == "ready"
    assert first["session"]["question_count"] == 1
    assert len(calls) == 1
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM llm_cache WHERE cache_key LIKE 'assessment_question_backfill:%'"
    ).fetchone()[0] == 2

    cached_learner = "owner:learner-b"
    _seed_completed_accuracy(
        conn,
        session_id="cached-backfill-history",
        correct_count=5,
        learner=cached_learner,
    )
    for index in range(2):
        service.record_scroll(
            conn, learner_id=cached_learner, reel_id=f"legacy-{index}"
        )
    cached = service.next_session(
        conn, learner_id=cached_learner, material_id=MATERIAL
    )
    assert cached["status"] == "ready"
    assert cached["session"]["question_count"] == 1
    assert len(calls) == 1

    conn.execute(
        "UPDATE llm_cache SET created_at = '2020-01-01T00:00:00+00:00' "
        "WHERE response_json LIKE '%null%'"
    )

    retry_learner = "owner:backfill-retry"
    _seed_completed_accuracy(
        conn,
        session_id="retry-backfill-history",
        correct_count=5,
        learner=retry_learner,
    )
    for index in range(2):
        service.record_scroll(
            conn, learner_id=retry_learner, reel_id=f"legacy-{index}"
        )
    retried = service.next_session(
        conn, learner_id=retry_learner, material_id=MATERIAL
    )
    assert len(calls) == 2
    assert retried["status"] == "ready"
    assert retried["session"]["question_count"] == 2
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 2


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
            "(learner_id, reel_id, material_id, max_fraction, scrolled_at, "
            "completed_at, created_at, updated_at) "
            "VALUES (?, ?, ?, 1.0, ?, ?, ?, ?)",
            (LEARNER, reel_id, MATERIAL, timestamp, timestamp, timestamp, timestamp),
        )
    rows = service._scrolled_rows(conn, LEARNER, MATERIAL)
    for row in rows[:12]:
        service._write_cached_backfill(conn, _question_fingerprint(row), None)
    service._ensure_question_pool(
        conn,
        learner_id=LEARNER,
        material_id=MATERIAL,
        source_rows=rows,
        should_cancel=None,
    )
    assert calls == [["starved-13", "starved-12"]]
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 2


def test_question_count_is_two_distinct_reels_with_one_reel_fallback(conn) -> None:
    service = AssessmentService()
    state = {
        "available_questions": [
            {"id": "q1", "reel_id": "r1", "concept_id": "c1"},
            {"id": "q1-duplicate", "reel_id": "r1", "concept_id": "c1"},
            {"id": "q2", "reel_id": "r2", "concept_id": "c2"},
        ],
    }
    assert service._desired_question_count(conn, state) == 2
    state["available_questions"] = [
        {"id": "q1", "reel_id": "r1", "concept_id": "c1"}
    ]
    assert service._desired_question_count(conn, state) == 1
    state["available_questions"] = [
        {"id": "q1", "reel_id": "r1", "concept_id": "c1"},
        {"id": "q2", "reel_id": "r2", "concept_id": "c1"},
    ]
    assert service._desired_question_count(conn, state) == 1


@pytest.mark.parametrize("winner_status", ["completed", "snoozed"])
def test_next_session_does_not_reopen_a_stale_cadence_window(
    conn, monkeypatch, winner_status
) -> None:
    _seed_completed_accuracy(conn, session_id="race-history", correct_count=5)
    service = AssessmentService()
    for index in range(2):
        _seed_reel(
            conn,
            reel_id=f"race-{index}",
            concept_id=f"race-c{index}",
            video_id=f"race-v{index}",
        )
        _complete(service, conn, f"race-{index}")

    original_execute = assessments_module.execute_modify
    injected_winner = False
    running_winner = False

    def interleaving_execute(connection, query, params=()):
        nonlocal injected_winner, running_winner
        if (
            "INSERT INTO assessment_sessions" in query
            and not injected_winner
            and not running_winner
        ):
            injected_winner = True
            running_winner = True
            try:
                winner = service.next_session(
                    connection, learner_id=LEARNER, material_id=MATERIAL
                )
                assert winner["status"] == "ready"
                session = winner["session"]
                if winner_status == "snoozed":
                    terminal = service.snooze(
                        connection, learner_id=LEARNER, session_id=session["id"]
                    )
                    assert terminal["status"] == "snoozed"
                else:
                    answer = None
                    for question in session["questions"]:
                        answer = service.answer(
                            connection,
                            learner_id=LEARNER,
                            session_id=session["id"],
                            question_id=question["id"],
                            choice_index=0,
                        )
                    assert answer is not None
                    assert answer["session"]["status"] == "completed"
            finally:
                running_winner = False
        return original_execute(connection, query, params)

    monkeypatch.setattr(assessments_module, "execute_modify", interleaving_execute)
    stale = service.next_session(conn, learner_id=LEARNER, material_id=MATERIAL)

    assert injected_winner is True
    assert stale["status"] == "not_ready"
    assert stale["assessment_ready"] is False
    assert stale["session"] is None
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_sessions WHERE status = 'pending'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_sessions WHERE status = ?",
        (winner_status,),
    ).fetchone()[0] >= 1


def test_session_creation_cancels_before_writes_and_rolls_back_child_failure(
    conn, monkeypatch
) -> None:
    _seed_completed_accuracy(conn, session_id="atomic-history", correct_count=5)
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
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_sessions WHERE status = 'pending'"
    ).fetchone()[0] == 0

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
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_sessions WHERE status = 'pending'"
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM assessment_session_questions").fetchone()[0] == 0


def test_default_backfill_forwards_the_cancellation_probe(monkeypatch) -> None:
    probe = lambda: False
    observed: dict[str, object] = {}

    def fake_llm_json(*args, **kwargs):
        observed["prompt"] = args[0]
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
    assert "prompt at most 16 words" in str(observed["prompt"])
    assert "each option at most 8 words" in str(observed["prompt"])
    assert "explanation one sentence and at most 24 words" in str(observed["prompt"])


def test_next_session_backfill_observes_cancellation_before_any_write(conn) -> None:
    _seed_completed_accuracy(conn, session_id="cancel-history", correct_count=5)
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
    service.record_scroll(conn, learner_id=LEARNER, reel_id="cancel-0")
    service.record_scroll(conn, learner_id=LEARNER, reel_id="cancel-1")
    assert cancelled is False
    with pytest.raises(AssessmentCancelledError):
        service.next_session(
            conn,
            learner_id=LEARNER,
            material_id=MATERIAL,
            should_cancel=lambda: cancelled,
        )
    assert conn.execute("SELECT COUNT(*) FROM reel_assessment_questions").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM llm_cache WHERE cache_key LIKE 'assessment_question_backfill:%'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM assessment_sessions WHERE status = 'pending'"
    ).fetchone()[0] == 0


def _seed_promotion_outcome(
    conn,
    *,
    index: int,
    correct: int = 3,
    adjustment: float = 0.08,
) -> None:
    concept_id = f"promotion-c{index}"
    _seed_reel(
        conn,
        reel_id=f"promotion-r{index}",
        concept_id=concept_id,
        video_id=f"promotion-v{index}",
        with_question=False,
    )
    session_id = f"promotion-session-{index}"
    created_at = f"2026-07-10T00:0{index}:00+00:00"
    _seed_completed_accuracy(
        conn,
        session_id=session_id,
        correct_count=correct,
        question_count=3,
        completed_at=created_at,
    )
    conn.execute(
        "INSERT INTO assessment_concept_outcomes "
        "(learner_id, session_id, material_id, concept_id, question_count, "
        "correct_count, accuracy, adjustment, created_at) "
        "VALUES (?, ?, ?, ?, 3, ?, ?, ?, ?)",
        (
            LEARNER,
            session_id,
            MATERIAL,
            concept_id,
            correct,
            correct / 3,
            adjustment,
            created_at,
        ),
    )


def test_auto_promotion_requires_sustained_broad_mastery_and_never_demotes(conn) -> None:
    service = AssessmentService()
    service._ensure_learner_progress(conn, LEARNER, MATERIAL)
    conn.execute(
        "UPDATE learner_material_progress SET difficulty_reset_at = ? "
        "WHERE learner_id = ? AND material_id = ?",
        ("2026-07-09T00:00:00+00:00", LEARNER, MATERIAL),
    )
    for index in range(3):
        _seed_promotion_outcome(conn, index=index)

    promoted = service._maybe_auto_promote_level(
        conn,
        learner_id=LEARNER,
        material_id=MATERIAL,
        promoted_at="2026-07-11T00:00:00+00:00",
    )
    assert promoted == "intermediate"
    progress = conn.execute(
        "SELECT selected_level, global_adjustment FROM learner_material_progress "
        "WHERE learner_id = ? AND material_id = ?",
        (LEARNER, MATERIAL),
    ).fetchone()
    assert tuple(progress) == ("intermediate", 0.0)

    conn.execute(
        "UPDATE learner_material_progress SET selected_level = 'advanced' "
        "WHERE learner_id = ? AND material_id = ?",
        (LEARNER, MATERIAL),
    )
    assert service._maybe_auto_promote_level(
        conn,
        learner_id=LEARNER,
        material_id=MATERIAL,
        promoted_at="2026-07-12T00:00:00+00:00",
    ) is None
    assert conn.execute(
        "SELECT selected_level FROM learner_material_progress "
        "WHERE learner_id = ? AND material_id = ?",
        (LEARNER, MATERIAL),
    ).fetchone()[0] == "advanced"


def test_auto_promotion_is_blocked_by_recent_negative_outcome(conn) -> None:
    service = AssessmentService()
    service._ensure_learner_progress(conn, LEARNER, MATERIAL)
    conn.execute(
        "UPDATE learner_material_progress SET difficulty_reset_at = ? "
        "WHERE learner_id = ? AND material_id = ?",
        ("2026-07-09T00:00:00+00:00", LEARNER, MATERIAL),
    )
    for index in range(3):
        _seed_promotion_outcome(
            conn,
            index=index,
            correct=2 if index == 2 else 3,
            adjustment=-0.12 if index == 2 else 0.08,
        )
    assert service._maybe_auto_promote_level(
        conn,
        learner_id=LEARNER,
        material_id=MATERIAL,
        promoted_at="2026-07-11T00:00:00+00:00",
    ) is None
    assert conn.execute(
        "SELECT selected_level FROM learner_material_progress "
        "WHERE learner_id = ? AND material_id = ?",
        (LEARNER, MATERIAL),
    ).fetchone()[0] == "beginner"


def test_auto_promotion_does_not_require_three_separate_sessions(conn) -> None:
    service = AssessmentService()
    service._ensure_learner_progress(conn, LEARNER, MATERIAL)
    conn.execute(
        "UPDATE learner_material_progress SET difficulty_reset_at = ? "
        "WHERE learner_id = ? AND material_id = ?",
        ("2026-07-09T00:00:00+00:00", LEARNER, MATERIAL),
    )
    for index in range(3):
        _seed_promotion_outcome(conn, index=index)
    conn.execute(
        "UPDATE assessment_concept_outcomes SET session_id = ? "
        "WHERE learner_id = ? AND material_id = ?",
        ("promotion-session-0", LEARNER, MATERIAL),
    )

    assert service._maybe_auto_promote_level(
        conn,
        learner_id=LEARNER,
        material_id=MATERIAL,
        promoted_at="2026-07-11T00:00:00+00:00",
    ) == "intermediate"
