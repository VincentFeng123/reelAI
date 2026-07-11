"""FastAPI assessment contract, privacy, resume, and learner isolation."""

from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

from backend.app import db as db_module
from backend.app.config import get_settings
import backend.app.main as main_module
from backend.app.services.assessments import store_reel_assessment_question


OWNER_HEADER = "x-studyreels-owner-key"
OWNER_A = "assessment-owner-a-abcdefghijklmnopqrstuvwxyz"
OWNER_B = "assessment-owner-b-abcdefghijklmnopqrstuvwxyz"
MATERIAL = "assessment-api-material"


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
            for index in range(3):
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
        first = self.client.post(
            "/api/reels/api-reel-0/progress",
            headers=self.headers_a,
            json={"max_fraction": 1.0},
        )
        second = self.client.post(
            "/api/reels/api-reel-1/progress",
            headers=self.headers_a,
            json={"max_fraction": 1.0},
        )
        third = self.client.post(
            "/api/reels/api-reel-2/progress",
            headers=self.headers_a,
            json={"max_fraction": 1.0},
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 200
        assert second.json()["assessment_ready"] is False
        assert third.json()["assessment_ready"] is True
        assert {
            "reel_id",
            "completed",
            "newly_completed",
            "assessment_ready",
            "information_units",
            "readiness_threshold",
        } == set(third.json())

        created = self.client.post(
            "/api/assessments/next",
            headers=self.headers_a,
            json={"material_id": MATERIAL},
        )
        assert created.status_code == 200
        body = created.json()
        assert body["status"] == "ready"
        assert body["session"]["question_count"] == 2
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
        assert len(result["session"]["understood_concepts"]) == 2
        with db_module.get_conn() as conn:
            progress = db_module.fetch_one(
                conn,
                "SELECT global_adjustment FROM learner_material_progress "
                "WHERE material_id = ?",
                (MATERIAL,),
            )
        assert progress is not None
        assert float(progress["global_adjustment"]) == 0.16

    def test_other_learner_cannot_resume_or_answer_session(self) -> None:
        for index in range(3):
            response = self.client.post(
                f"/api/reels/api-reel-{index}/progress",
                headers=self.headers_a,
                json={"max_fraction": 1.0},
            )
            assert response.status_code == 200
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

    def test_progress_requires_a_real_study_reel(self) -> None:
        response = self.client.post(
            "/api/reels/community-only/progress",
            headers=self.headers_a,
            json={"max_fraction": 1.0},
        )
        assert response.status_code == 404
