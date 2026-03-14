import sqlite3
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.app.main as main_module
from backend.app.main import _resolve_target_clip_duration_bounds
from backend.app.db import _migrate_reels_unique_clip_index_sqlite
from backend.app.services.material_intelligence import MaterialIntelligenceService
from backend.app.services.reels import QueryCandidate, ReelService
from backend.app.services.youtube import YouTubeService
from backend.app.main import _build_generation_request_key


class MediumRegressionTests(unittest.TestCase):
    def _build_generation_test_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE materials (
                id TEXT PRIMARY KEY
            );

            CREATE TABLE reel_generations (
                id TEXT PRIMARY KEY,
                material_id TEXT NOT NULL,
                concept_id TEXT,
                request_key TEXT NOT NULL,
                generation_mode TEXT NOT NULL DEFAULT 'fast',
                retrieval_profile TEXT NOT NULL DEFAULT 'bootstrap',
                status TEXT NOT NULL DEFAULT 'pending',
                source_generation_id TEXT,
                reel_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                activated_at TEXT,
                error_text TEXT
            );

            CREATE TABLE reel_generation_heads (
                id TEXT PRIMARY KEY,
                material_id TEXT NOT NULL,
                request_key TEXT NOT NULL,
                active_generation_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE reel_generation_jobs (
                id TEXT PRIMARY KEY,
                material_id TEXT NOT NULL,
                concept_id TEXT,
                request_key TEXT NOT NULL,
                source_generation_id TEXT NOT NULL,
                result_generation_id TEXT,
                target_profile TEXT NOT NULL DEFAULT 'deep',
                request_params_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_text TEXT
            );

            CREATE TABLE reels (
                id TEXT PRIMARY KEY,
                generation_id TEXT,
                material_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                video_url TEXT NOT NULL DEFAULT '',
                t_start REAL NOT NULL,
                t_end REAL NOT NULL,
                transcript_snippet TEXT NOT NULL DEFAULT '',
                takeaways_json TEXT NOT NULL DEFAULT '[]',
                base_score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.execute("INSERT INTO materials (id) VALUES (?)", ("material-1",))
        return conn

    def _fake_ranked_request_reels(self, conn: sqlite3.Connection, generation_id: str | None) -> list[dict[str, object]]:
        if generation_id:
            rows = conn.execute(
                "SELECT id FROM reels WHERE generation_id = ? ORDER BY created_at ASC, id ASC",
                (generation_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id FROM reels WHERE generation_id IS NULL ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [
            {
                "reel_id": str(row["id"]),
                "video_duration_sec": 600,
                "clip_duration_sec": 55.0,
            }
            for row in rows
        ]

    def test_main_clip_bounds_respect_minimum_duration(self) -> None:
        self.assertEqual(
            _resolve_target_clip_duration_bounds(15, None, None),
            (15, 15, 30),
        )

    def test_reel_service_clip_bounds_respect_minimum_duration(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        self.assertEqual(
            service._resolve_clip_duration_bounds(15, None, None),
            (15, 30, 15),
        )

    def test_material_intelligence_cache_key_uses_full_text(self) -> None:
        service = MaterialIntelligenceService()
        shared_prefix = "a" * 24_000
        key_a = service._cache_key(shared_prefix + "first-suffix", "systems", 12)
        key_b = service._cache_key(shared_prefix + "second-suffix", "systems", 12)
        self.assertNotEqual(key_a, key_b)

    def test_generation_request_key_changes_with_generation_settings(self) -> None:
        base = _build_generation_request_key(
            material_id="material-1",
            concept_id=None,
            creative_commons_only=False,
            generation_mode="fast",
            video_pool_mode="short-first",
            preferred_video_duration="any",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )
        changed = _build_generation_request_key(
            material_id="material-1",
            concept_id=None,
            creative_commons_only=False,
            generation_mode="fast",
            video_pool_mode="balanced",
            preferred_video_duration="any",
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )
        self.assertNotEqual(base, changed)

    def test_ensure_generation_queues_refinement_when_bootstrap_is_sufficient(self) -> None:
        conn = self._build_generation_test_conn()
        generate_calls: list[str] = []

        def fake_generate_reels(*args, **kwargs):
            generation_id = str(kwargs["generation_id"])
            material_id = str(kwargs["material_id"])
            retrieval_profile = str(kwargs["retrieval_profile"])
            num_reels = int(kwargs["num_reels"])
            generate_calls.append(retrieval_profile)
            for index in range(num_reels):
                conn.execute(
                    """
                    INSERT INTO reels (
                        id, generation_id, material_id, concept_id, video_id, video_url,
                        t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, '', 0, 55, '', '[]', 0, ?)
                    """,
                    (
                        f"{generation_id}-{retrieval_profile}-{index}",
                        generation_id,
                        material_id,
                        "concept-1",
                        f"video-{retrieval_profile}-{index}",
                        f"2026-03-13T00:00:{index:02d}+00:00",
                    ),
                )
            return []

        with mock.patch.object(main_module.reel_service, "generate_reels", side_effect=fake_generate_reels), mock.patch.object(
            main_module,
            "_ranked_request_reels",
            side_effect=lambda test_conn, **kwargs: self._fake_ranked_request_reels(test_conn, kwargs["generation_id"]),
        ), mock.patch.object(
            main_module,
            "_queue_refinement_job",
            return_value={"id": "job-1", "status": "queued"},
        ) as queue_job:
            result = main_module._ensure_generation_for_request(
                conn,
                material_id="material-1",
                concept_id=None,
                required_count=2,
                creative_commons_only=False,
                generation_mode="fast",
                min_relevance=None,
                video_pool_mode="short-first",
                preferred_video_duration="any",
                target_clip_duration_sec=55,
                target_clip_duration_min_sec=20,
                target_clip_duration_max_sec=55,
            )

        self.assertEqual(generate_calls, ["bootstrap"])
        self.assertEqual(result["response_profile"], "bootstrap")
        self.assertEqual(result["refinement_job_id"], "job-1")
        self.assertEqual(result["refinement_status"], "queued")
        self.assertEqual(len(result["reels"]), 2)
        queue_job.assert_called_once()
        head = conn.execute("SELECT active_generation_id FROM reel_generation_heads").fetchone()
        self.assertIsNotNone(head)
        self.assertEqual(str(head["active_generation_id"]), str(result["generation_id"]))
        generation_row = conn.execute(
            "SELECT retrieval_profile, status FROM reel_generations WHERE id = ?",
            (result["generation_id"],),
        ).fetchone()
        self.assertEqual((generation_row["retrieval_profile"], generation_row["status"]), ("bootstrap", "active"))
        conn.close()

    def test_ensure_generation_runs_sync_deep_fallback_when_bootstrap_is_under_target(self) -> None:
        conn = self._build_generation_test_conn()
        generate_calls: list[tuple[str, int]] = []

        def fake_generate_reels(*args, **kwargs):
            generation_id = str(kwargs["generation_id"])
            material_id = str(kwargs["material_id"])
            retrieval_profile = str(kwargs["retrieval_profile"])
            requested = int(kwargs["num_reels"])
            generate_calls.append((retrieval_profile, requested))
            insert_count = 1 if retrieval_profile == "bootstrap" else requested
            existing_count = conn.execute(
                "SELECT COUNT(*) FROM reels WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()[0]
            for index in range(insert_count):
                sequence = existing_count + index
                conn.execute(
                    """
                    INSERT INTO reels (
                        id, generation_id, material_id, concept_id, video_id, video_url,
                        t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, '', 0, 55, '', '[]', 0, ?)
                    """,
                    (
                        f"{generation_id}-{retrieval_profile}-{sequence}",
                        generation_id,
                        material_id,
                        "concept-1",
                        f"video-{retrieval_profile}-{sequence}",
                        f"2026-03-13T00:00:{sequence:02d}+00:00",
                    ),
                )
            return []

        with mock.patch.object(main_module.reel_service, "generate_reels", side_effect=fake_generate_reels), mock.patch.object(
            main_module,
            "_ranked_request_reels",
            side_effect=lambda test_conn, **kwargs: self._fake_ranked_request_reels(test_conn, kwargs["generation_id"]),
        ), mock.patch.object(main_module, "_queue_refinement_job") as queue_job:
            result = main_module._ensure_generation_for_request(
                conn,
                material_id="material-1",
                concept_id=None,
                required_count=3,
                creative_commons_only=False,
                generation_mode="fast",
                min_relevance=None,
                video_pool_mode="short-first",
                preferred_video_duration="any",
                target_clip_duration_sec=55,
                target_clip_duration_min_sec=20,
                target_clip_duration_max_sec=55,
            )

        self.assertEqual(generate_calls, [("bootstrap", 3), ("deep", 2)])
        self.assertEqual(result["response_profile"], "bootstrap_then_deep")
        self.assertIsNone(result["refinement_job_id"])
        self.assertIsNone(result["refinement_status"])
        self.assertEqual(len(result["reels"]), 3)
        queue_job.assert_not_called()
        generation_row = conn.execute(
            "SELECT retrieval_profile, status, reel_count FROM reel_generations WHERE id = ?",
            (result["generation_id"],),
        ).fetchone()
        self.assertEqual((generation_row["retrieval_profile"], generation_row["status"]), ("bootstrap_then_deep", "active"))
        self.assertEqual(int(generation_row["reel_count"]), 3)
        conn.close()

    def test_reel_unique_clip_migration_allows_same_clip_across_generations(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                """
                CREATE TABLE reels (
                    id TEXT PRIMARY KEY,
                    generation_id TEXT,
                    material_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    video_url TEXT NOT NULL DEFAULT '',
                    t_start REAL NOT NULL,
                    t_end REAL NOT NULL,
                    transcript_snippet TEXT NOT NULL DEFAULT '',
                    takeaways_json TEXT NOT NULL DEFAULT '[]',
                    base_score REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO reels (
                    id, generation_id, material_id, concept_id, video_id, video_url,
                    t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
                ) VALUES (?, ?, ?, ?, ?, '', ?, ?, '', '[]', 0, '')
                """,
                [
                    ("r1", "gen-a", "material-1", "concept-1", "video-1", 0.0, 10.0),
                    ("r2", "gen-b", "material-1", "concept-1", "video-1", 0.0, 10.0),
                    ("r3", "gen-b", "material-1", "concept-1", "video-1", 0.0, 10.0),
                ],
            )

            _migrate_reels_unique_clip_index_sqlite(conn)

            rows = conn.execute(
                "SELECT generation_id, COUNT(*) AS reel_count FROM reels GROUP BY generation_id ORDER BY generation_id"
            ).fetchall()
            self.assertEqual([(row[0], row[1]) for row in rows], [("gen-a", 1), ("gen-b", 1)])
        finally:
            conn.close()

    def test_reel_service_bootstrap_query_candidates_are_capped_and_html_first(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)

        candidates = service._build_query_candidates(
            title="Binary Search Trees",
            keywords=["algorithms", "tree traversal", "data structures"],
            summary="Study how BST insertion, search, and traversal work.",
            subject_tag="computer science",
            context_terms=["lecture", "study guide"],
            visual_spec={
                "environment": ["whiteboard", "diagram"],
                "subjects": ["tree"],
                "objects": ["nodes"],
                "actions": ["explaining"],
            },
            fast_mode=False,
            retrieval_profile="bootstrap",
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(sum(1 for candidate in candidates if candidate.stage == "high_precision"), 2)
        self.assertEqual(sum(1 for candidate in candidates if candidate.stage == "recovery"), 1)
        self.assertTrue(all(candidate.source_surface == "youtube_html" for candidate in candidates))

    def test_reel_service_bootstrap_stage_plan_skips_broad(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        candidates = [
            QueryCandidate("query one", "literal", 0.9, stage="high_precision", source_surface="youtube_html"),
            QueryCandidate("query two", "scene", 0.8, stage="high_precision", source_surface="youtube_html"),
            QueryCandidate("query three", "recovery_adjacent", 0.7, stage="recovery", source_surface="youtube_html"),
        ]

        plans = service._build_retrieval_stage_plan(
            query_candidates=candidates,
            fast_mode=False,
            retrieval_profile="bootstrap",
        )

        self.assertEqual([plan.name for plan in plans], ["high_precision", "recovery"])
        self.assertEqual(plans[0].budget, 2)
        self.assertEqual(plans[1].budget, 1)

    def test_reel_service_bootstrap_duration_plan_uses_single_none(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        self.assertEqual(
            service._stage_duration_plan(
                stage_name="high_precision",
                preferred_video_duration="short",
                video_pool_mode="balanced",
                fast_mode=False,
                retrieval_profile="bootstrap",
            ),
            (None,),
        )

    def test_reel_service_bootstrap_weak_pool_prefers_score_and_diversity(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)

        weak_candidates = [
            {
                "ranking": {"final_score": 0.3},
                "video": {"channel_title": "Channel A"},
                "query_candidate": QueryCandidate("q1", "literal", 0.9),
            },
            {
                "ranking": {"final_score": 0.28},
                "video": {"channel_title": "Channel A"},
                "query_candidate": QueryCandidate("q2", "literal", 0.85),
            },
        ]
        self.assertTrue(service._bootstrap_pool_is_weak(weak_candidates, max_generation_target=2))

        strong_candidates = [
            {
                "ranking": {"final_score": 0.35},
                "video": {"channel_title": "Channel A"},
                "query_candidate": QueryCandidate("q1", "literal", 0.9),
            },
            {
                "ranking": {"final_score": 0.31},
                "video": {"channel_title": "Channel B"},
                "query_candidate": QueryCandidate("q2", "scene", 0.85),
            },
            {
                "ranking": {"final_score": 0.29},
                "video": {"channel_title": "Channel C"},
                "query_candidate": QueryCandidate("q3", "recovery_adjacent", 0.8),
            },
        ]
        self.assertFalse(service._bootstrap_pool_is_weak(strong_candidates, max_generation_target=2))

    def test_youtube_service_bootstrap_query_variants_use_single_html_query(self) -> None:
        service = YouTubeService()
        variants = service._build_search_query_variants(
            query="binary search trees",
            video_duration="short",
            source_surface="youtube_html",
            retrieval_profile="bootstrap",
        )

        self.assertEqual(variants, [{"query": "binary search trees", "surface": "youtube_html"}])

    def test_youtube_service_bootstrap_creative_commons_uses_data_api_when_available(self) -> None:
        service = YouTubeService()
        service.api_key = "test-key"

        html_calls: list[str] = []
        data_api_calls: list[str] = []

        def fake_search_via_data_api(*, query: str, **_: object) -> list[dict[str, object]]:
            data_api_calls.append(query)
            return [{"id": "abc123xyz00", "title": "CC video"}]

        def fake_search_without_data_api(*, query: str, **_: object) -> list[dict[str, object]]:
            html_calls.append(query)
            return []

        service._search_via_data_api = fake_search_via_data_api  # type: ignore[method-assign]
        service._search_without_data_api = fake_search_without_data_api  # type: ignore[method-assign]

        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE search_cache (cache_key TEXT PRIMARY KEY, response_json TEXT, created_at TEXT)"
            )
            rows = service._search_videos_with_conn(
                conn,
                query="binary search trees",
                max_results=4,
                creative_commons_only=True,
                video_duration=None,
                retrieval_strategy="literal",
                retrieval_stage="high_precision",
                source_surface="youtube_html",
                retrieval_profile="bootstrap",
                allow_external_fallbacks=False,
                variant_limit=1,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(data_api_calls, ["binary search trees"])
        self.assertEqual(html_calls, [])


if __name__ == "__main__":
    unittest.main()
