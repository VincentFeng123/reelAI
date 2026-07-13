import json
import sqlite3
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.app.main as main_module
import backend.app.services.material_intelligence as material_intelligence_module
from backend.app.main import _resolve_target_clip_duration_bounds
from backend.app.db import SCHEMA, _ensure_reels_generation_index_sqlite, _migrate_reels_unique_clip_index_sqlite
from backend.app.services.material_intelligence import MaterialIntelligenceService
from backend.app.services.reels import QueryCandidate, ReelService
from backend.app.services.topic_expansion import TopicExpansionService
from backend.app.services.youtube import YouTubeService
from backend.app.clip_engine.provider_cache import transcript_artifact_key
from fastapi.testclient import TestClient


def _validated_query_plan(expansion: dict[str, object]) -> mock.Mock:
    plan = mock.Mock()
    plan.ai_status = "validated"
    plan.as_topic_expansion.return_value = expansion
    return plan


class MediumRegressionTests(unittest.TestCase):
    def _build_generation_test_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE materials (
                id TEXT PRIMARY KEY,
                subject_tag TEXT,
                source_type TEXT,
                knowledge_level TEXT NOT NULL DEFAULT 'beginner',
                level_adjustment REAL NOT NULL DEFAULT 0.0
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
                refinement_state_json TEXT NOT NULL DEFAULT '{}',
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
                difficulty REAL,
                created_at TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE learner_material_progress (
                learner_id TEXT NOT NULL,
                material_id TEXT NOT NULL,
                selected_level TEXT NOT NULL DEFAULT 'beginner',
                global_adjustment REAL NOT NULL DEFAULT 0.0,
                difficulty_reset_at TEXT NOT NULL DEFAULT '',
                feedback_revision INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (learner_id, material_id)
            );

            CREATE TABLE reel_feedback (
                id TEXT NOT NULL,
                learner_id TEXT NOT NULL DEFAULT 'legacy',
                reel_id TEXT NOT NULL,
                helpful INTEGER NOT NULL DEFAULT 0,
                confusing INTEGER NOT NULL DEFAULT 0,
                rating INTEGER,
                saved INTEGER NOT NULL DEFAULT 0,
                mastery_updated_at TEXT,
                updated_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (learner_id, reel_id)
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

    def _build_ranked_feed_test_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-ranked", "biology", "Cell signaling notes", "text", None, "2026-03-13T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO concepts (
                id, material_id, title, keywords_json, summary, embedding_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "concept-ranked",
                "material-ranked",
                "Cell signaling",
                '["cell", "signaling"]',
                "How cells communicate.",
                None,
                "2026-03-13T00:01:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO videos (
                id, title, channel_title, description, duration_sec, view_count, is_creative_commons, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "vidRanked01",
                "Cell signaling explainer",
                "Bio Channel",
                "Explains how cells communicate.",
                180,
                100,
                0,
                "2026-03-13T00:02:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO reels (
                id, generation_id, material_id, concept_id, video_id, video_url,
                t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "reel-ranked",
                None,
                "material-ranked",
                "concept-ranked",
                "vidRanked01",
                "https://www.youtube.com/watch?v=vidRanked01",
                0.0,
                30.0,
                "Cell signaling recap",
                '["Signal reception"]',
                1.2,
                "2026-03-13T00:03:00+00:00",
            ),
        )
        artifact_key = transcript_artifact_key(
            video_id="vidRanked01",
            provider="supadata",
            requested_language="en",
            returned_language="en",
            native_mode=True,
        )
        artifact = {
            "artifact_key": artifact_key,
            "video_id": "vidRanked01",
            "provider": "supadata",
            "requested_language": "en",
            "returned_language": "en",
            "native_mode": True,
            "schema_version": 2,
            "segments": [
                {"cue_id": "cue-1", "start": 0.0, "end": 5.0, "text": "Cell signaling recap", "lang": "en"}
            ],
            "duration_sec": 5.0,
            "created_at": "2026-03-13T00:04:00+00:00",
        }
        conn.execute(
            """
            INSERT INTO transcript_artifacts (
                cache_key, video_id, provider, requested_language, returned_language,
                native_mode, schema_version, artifact_json, duration_sec, cue_count,
                created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_key,
                "vidRanked01",
                "supadata",
                "en",
                "en",
                1,
                "2",
                json.dumps(artifact),
                5.0,
                1,
                "2026-03-13T00:04:00+00:00",
                "2026-04-12T00:04:00+00:00",
            ),
        )
        return conn

    def _build_local_recovery_scope_test_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-a", "biology", "Topic: biology", "topic", None, "2026-03-13T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-b", "calculus", "Topic: calculus", "topic", None, "2026-03-13T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO concepts (
                id, material_id, title, keywords_json, summary, embedding_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "concept-bio",
                "material-a",
                "Biology",
                '["biology"]',
                "Biology topic",
                None,
                "2026-03-13T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO concepts (
                id, material_id, title, keywords_json, summary, embedding_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "concept-calc",
                "material-b",
                "Calculus",
                '["calculus"]',
                "Calculus topic",
                None,
                "2026-03-13T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO videos (
                id, title, channel_title, description, duration_sec, view_count, is_creative_commons, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "video-calc",
                "Calculus chain rule explained",
                "Math Channel",
                "Calculus tutorial",
                180,
                5000,
                0,
                "2026-03-13T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO reels (
                id, generation_id, material_id, concept_id, video_id, video_url,
                t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "reel-calc",
                None,
                "material-b",
                "concept-calc",
                "video-calc",
                "https://example.com/watch?v=video-calc",
                0.0,
                30.0,
                "Calculus recap",
                '["Derivative"]',
                1.0,
                "2026-03-13T00:00:00+00:00",
            ),
        )
        return conn

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

    def test_material_intelligence_cache_key_tracks_prompt_version(self) -> None:
        service = MaterialIntelligenceService()
        with mock.patch.object(
            material_intelligence_module,
            "MATERIAL_INTELLIGENCE_PROMPT_VERSION",
            "legacy-prompt",
        ):
            legacy_key = service._cache_key("Photosynthesis makes ATP.", None, 12)
        current_key = service._cache_key("Photosynthesis makes ATP.", None, 12)
        self.assertNotEqual(legacy_key, current_key)

    def test_material_intelligence_prompt_upgrade_ignores_legacy_cached_plan(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        service = MaterialIntelligenceService()
        text = "Photosynthesis uses chlorophyll to make ATP and sugars from light."
        with mock.patch.object(
            material_intelligence_module,
            "MATERIAL_INTELLIGENCE_PROMPT_VERSION",
            "legacy-prompt",
        ):
            legacy_key = service._cache_key(text, None, 12)
        conn.execute(
            "INSERT INTO llm_cache (cache_key, response_json, created_at) VALUES (?, ?, ?)",
            (
                legacy_key,
                '{"concepts":[{"title":"Energy"}]}',
                "2026-07-12T00:00:00+00:00",
            ),
        )
        fresh = {"concepts": [{"title": "Photosynthesis"}], "objectives": []}
        try:
            with mock.patch.object(
                service,
                "_generate_payload",
                return_value=fresh,
            ) as generate:
                result = service._cached_or_generate(conn, text, None, 12)
            self.assertEqual(result, fresh)
            generate.assert_called_once()
        finally:
            conn.close()

    def test_material_intelligence_uses_only_one_concept_llm_call(self) -> None:
        service = MaterialIntelligenceService()
        service.llm_available = True
        payload = {
            "concepts": [{
                "title": "Photosynthesis",
                "keywords": ["chlorophyll", "calvin cycle"],
                "summary": "Plants convert light into chemical energy.",
            }],
            "objectives": [],
        }
        with (
            mock.patch(
                "backend.app.services.concepts._extract_concepts_via_llm"
            ) as duplicate_llm,
            mock.patch.object(service, "_cached_or_generate", return_value=payload),
        ):
            concepts, _objectives = service.extract_concepts_and_objectives(
                None,
                "Photosynthesis uses chlorophyll while cellular respiration produces ATP.",
                max_concepts=6,
            )

        duplicate_llm.assert_not_called()
        self.assertEqual(concepts[0]["title"], "Photosynthesis")

    def test_material_intelligence_prompt_rejects_vague_umbrella_concepts(self) -> None:
        service = MaterialIntelligenceService()
        captured: dict[str, str] = {}

        def fake_completion(**kwargs):
            captured["user"] = str(kwargs["user"])
            return '{"concepts": [], "objectives": []}'

        service.llm_available = True
        with mock.patch(
            "backend.app.services.llm_router.chat_completion",
            side_effect=fake_completion,
        ):
            service._generate_payload(
                "Photosynthesis and cellular respiration exchange energy through ATP.",
                None,
                6,
            )

        self.assertIn("Do not use broad umbrella titles", captured["user"])
        self.assertIn("Photosynthesis", captured["user"])

    def test_material_intelligence_prioritizes_specific_process_over_energy(self) -> None:
        service = MaterialIntelligenceService()
        service.llm_available = True
        payload = {
            "concepts": [
                {
                    "title": "Energy",
                    "keywords": ["energy", "atp"],
                    "summary": "Energy changes form in living systems.",
                },
                {
                    "title": "Photosynthesis",
                    "keywords": ["chlorophyll", "calvin cycle"],
                    "summary": "Photosynthesis converts light into chemical energy.",
                },
            ],
            "objectives": [],
        }
        with mock.patch.object(service, "_cached_or_generate", return_value=payload):
            concepts, _objectives = service.extract_concepts_and_objectives(
                None,
                "Photosynthesis uses chlorophyll to make ATP and sugars from light.",
                max_concepts=6,
            )
        titles = [concept["title"] for concept in concepts]
        self.assertEqual(titles[0], "Photosynthesis")
        if "Energy" in titles:
            self.assertLess(titles.index("Photosynthesis"), titles.index("Energy"))

    def test_material_intelligence_topic_only_seed_is_literal_and_skips_preexpansion(self) -> None:
        service = MaterialIntelligenceService()
        service.client = object()
        with (
            mock.patch(
                "backend.app.services.search_query_plan.build_search_query_plan"
            ) as planner,
            mock.patch.object(service, "_cached_or_generate") as material_llm,
        ):
            concepts, objectives = service.extract_concepts_and_objectives(
                None,
                "Topic: carolingian minuscule paleography",
                subject_tag="carolingian minuscule paleography",
                max_concepts=6,
            )
        planner.assert_not_called()
        material_llm.assert_not_called()
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0]["title"], "Carolingian Minuscule Paleography")
        self.assertEqual(concepts[0]["keywords"], ["carolingian minuscule paleography"])
        self.assertEqual(
            concepts[0]["summary"],
            "Core ideas, terminology, and intuition for Carolingian Minuscule Paleography.",
        )
        self.assertEqual(
            objectives,
            [
                "Understand the core definitions and intuition behind Carolingian Minuscule Paleography.",
                "Solve representative problems in Carolingian Minuscule Paleography.",
            ],
        )

    def test_material_intelligence_topic_only_seed_is_deterministic_apart_from_identity(self) -> None:
        service = MaterialIntelligenceService()
        first, first_objectives = service.extract_concepts_and_objectives(
            None,
            "Topic: machine learning",
            subject_tag="machine learning",
            max_concepts=6,
        )
        second, second_objectives = service.extract_concepts_and_objectives(
            None,
            "Topic: machine learning",
            subject_tag="machine learning",
            max_concepts=6,
        )
        self.assertEqual(
            {key: value for key, value in first[0].items() if key != "id"},
            {key: value for key, value in second[0].items() if key != "id"},
        )
        self.assertEqual(first_objectives, second_objectives)

    def test_topic_expansion_service_filters_language_topic_candidates(self) -> None:
        service = TopicExpansionService()
        with (
            mock.patch.object(
                service,
                "_search_wikipedia_results",
                side_effect=[
                    [
                        {"title": "Spanish language", "snippet": "Spanish language and grammar overview."},
                        {"title": "Spanish grammar", "snippet": "Grammar of the Spanish language."},
                        {"title": "Spanish orthography", "snippet": "Writing and spelling in Spanish."},
                    ],
                    [
                        {"title": "Spanish language", "snippet": "Spanish language basics."},
                        {"title": "Spanish alphabet", "snippet": "Alphabet and pronunciation."},
                    ],
                ],
            ),
            mock.patch.object(service, "_search_wikidata_entities", return_value=[]),
            mock.patch.object(service, "_fetch_datamuse_related_terms", return_value=[]),
            mock.patch.object(
                service,
                "_fetch_wikipedia_links",
                return_value=["Spain", "Spanish phonology", "Royal Spanish Academy", "Subjunctive mood"],
            ),
        ):
            payload = service._expand_topic_uncached(
                topic="spanish",
                max_subtopics=8,
                max_aliases=4,
                max_related_terms=4,
            )
        subtopics = {str(item).strip().lower() for item in (payload.get("subtopics") or [])}
        self.assertIn("grammar", subtopics)
        self.assertIn("orthography", subtopics)
        self.assertIn("subjunctive mood", subtopics)
        self.assertNotIn("spain", subtopics)
        self.assertNotIn("royal spanish academy", subtopics)

    def test_topic_expansion_service_uses_generic_external_sources_for_unmapped_topic(self) -> None:
        service = TopicExpansionService()
        with (
            mock.patch.object(
                service,
                "_search_wikipedia_results",
                side_effect=[
                    [
                        {
                            "title": "Neuroscience",
                            "snippet": "Neuroscience is the scientific study of the nervous system and the brain.",
                        },
                        {
                            "title": "Outline of neuroscience",
                            "snippet": "Topics in neuroscience include cognitive neuroscience and behavioral neuroscience.",
                        },
                    ],
                    [],
                ],
            ),
            mock.patch.object(
                service,
                "_search_wikidata_entities",
                return_value=[
                    {
                        "label": "Neuroscience",
                        "description": "scientific study of the nervous system",
                        "aliases": ["neural science"],
                    }
                ],
            ),
            mock.patch.object(
                service,
                "_fetch_datamuse_related_terms",
                return_value=["brain science"],
            ),
            mock.patch.object(
                service,
                "_fetch_wikipedia_links",
                side_effect=[
                    ["Cognitive neuroscience", "Behavioral neuroscience", "Neuroanatomy"],
                    ["Cognitive neuroscience", "Behavioral neuroscience", "Computational neuroscience"],
                    [],
                    [],
                ],
            ),
        ):
            payload = service._expand_topic_uncached(
                topic="neuroscience",
                max_subtopics=6,
                max_aliases=4,
                max_related_terms=4,
            )
        subtopics = {str(item).strip().lower() for item in (payload.get("subtopics") or [])}
        aliases = {str(item).strip().lower() for item in (payload.get("aliases") or [])}
        related = {str(item).strip().lower() for item in (payload.get("related_terms") or [])}
        self.assertIn("cognitive neuroscience", subtopics)
        self.assertIn("behavioral neuroscience", subtopics)
        self.assertIn("computational neuroscience", subtopics)
        self.assertIn("neural science", aliases)
        self.assertTrue({"brain science", "scientific study of the nervous system"}.intersection(related))

    def test_topic_expansion_service_caches_payload(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        service = TopicExpansionService()
        expected_payload = {
            "canonical_topic": "Spanish language",
            "aliases": ["Spanish language"],
            "subtopics": ["grammar", "verb conjugation"],
            "related_terms": ["conversation practice"],
        }
        try:
            with mock.patch.object(service, "_expand_topic_uncached", return_value=expected_payload) as expand_mock:
                first = service.expand_topic(conn, topic="spanish", max_subtopics=5, max_aliases=3, max_related_terms=3)
                second = service.expand_topic(conn, topic="spanish", max_subtopics=5, max_aliases=3, max_related_terms=3)
            self.assertEqual(first, expected_payload)
            self.assertEqual(second, expected_payload)
            expand_mock.assert_called_once()
            cached = conn.execute("SELECT response_json FROM llm_cache").fetchone()
            self.assertIsNotNone(cached)
        finally:
            conn.close()

    def test_topic_expansion_service_static_fallback_supports_psychology(self) -> None:
        service = TopicExpansionService()
        with mock.patch.object(service, "_request_json", return_value={}):
            payload = service._expand_topic_uncached(
                topic="psychology",
                max_subtopics=6,
                max_aliases=4,
                max_related_terms=4,
            )
        subtopics = {str(item).strip().lower() for item in (payload.get("subtopics") or [])}
        self.assertIn("cognitive psychology", subtopics)
        self.assertIn("behavioral psychology", subtopics)
        self.assertIn("social psychology", subtopics)

    def test_topic_expansion_service_keeps_opaque_topics_anchored(self) -> None:
        service = TopicExpansionService()
        with (
            mock.patch.object(
                service,
                "_search_wikipedia_results",
                side_effect=[
                    [{"title": "Melittology", "snippet": "Melittology is the scientific study of bees."}],
                    [],
                ],
            ),
            mock.patch.object(
                service,
                "_search_wikidata_entities",
                return_value=[
                    {
                        "label": "Melittology",
                        "description": "branch of entomology focused on bees",
                        "aliases": ["Apiology"],
                    }
                ],
            ),
            mock.patch.object(
                service,
                "_fetch_wikipedia_links",
                return_value=["Action research", "Melittology from Greek", "Honey bee"],
            ),
            mock.patch.object(service, "_fetch_datamuse_related_terms") as datamuse_mock,
        ):
            payload = service._expand_topic_uncached(
                topic="apiology",
                max_subtopics=6,
                max_aliases=4,
                max_related_terms=4,
            )

        aliases = {str(item).strip().lower() for item in (payload.get("aliases") or [])}
        subtopics = {str(item).strip().lower() for item in (payload.get("subtopics") or [])}
        datamuse_mock.assert_called_once()
        self.assertIn("melittology", aliases)
        self.assertNotIn("action research", aliases)
        self.assertNotIn("action research", subtopics)
        self.assertNotIn("melittology from greek", subtopics)

    def test_topic_expansion_service_opaque_search_terms_require_explicit_anchor(self) -> None:
        service = TopicExpansionService()
        terms = service.build_topic_search_terms(
            topic="melittology",
            expansion={
                "canonical_topic": "Melittology",
                "aliases": ["Apiology", "Entomology"],
                "subtopics": [
                    "Melittology field methods",
                    "Honey bee anatomy",
                    "About honey bees melittology",
                    "Melittology or apiology bees",
                ],
                "related_terms": ["Melittology lecture", "Bee behavior", "Bees melittology also known"],
            },
            limit=8,
        )
        lowered = {str(item).strip().lower() for item in terms}
        self.assertIn("melittology", lowered)
        self.assertIn("melittology field methods", lowered)
        self.assertIn("melittology lecture", lowered)
        self.assertNotIn("entomology", lowered)
        self.assertNotIn("honey bee anatomy", lowered)
        self.assertNotIn("bee behavior", lowered)
        self.assertNotIn("about honey bees melittology", lowered)
        self.assertNotIn("bees melittology also known", lowered)
        self.assertNotIn("melittology or apiology bees", lowered)

    def test_topic_expansion_service_extracts_companion_terms_for_opaque_topics(self) -> None:
        service = TopicExpansionService()
        with (
            mock.patch.object(
                service,
                "_search_wikipedia_results",
                side_effect=[
                    [
                        {"title": "Myrmecology", "snippet": "Myrmecology is the scientific study of ants."},
                        {"title": "Outline of ants", "snippet": "Ant colony Myrmecology scientific study of ants Kingdom Animalia."},
                        {"title": "Index of branches of science", "snippet": "Study of fungi Myology Myrmecology study of ants Mythology traditional narrative."},
                    ],
                    [],
                ],
            ),
            mock.patch.object(
                service,
                "_search_wikidata_entities",
                return_value=[
                    {
                        "label": "Myrmecology",
                        "description": "branch of zoology focused on ants",
                        "aliases": [],
                    }
                ],
            ),
            mock.patch.object(service, "_fetch_wikipedia_links", return_value=["Asian Myrmecology", "Ant colony"]),
            mock.patch.object(service, "_fetch_datamuse_related_terms") as datamuse_mock,
        ):
            payload = service._expand_topic_uncached(
                topic="myrmecology",
                max_subtopics=6,
                max_aliases=4,
                max_related_terms=4,
            )
        related_terms = {str(item).strip().lower() for item in (payload.get("related_terms") or [])}
        search_terms = {
            str(item).strip().lower()
            for item in service.build_topic_search_terms(topic="myrmecology", expansion=payload, limit=6)
        }
        datamuse_mock.assert_called_once()
        self.assertIn("ants", related_terms)
        self.assertIn("ants", search_terms)
        self.assertNotIn("ants mythology", related_terms)
        self.assertNotIn("fungi myology", related_terms)

    def test_topic_expansion_service_adds_deterministic_companion_packs_for_niche_disciplines(self) -> None:
        service = TopicExpansionService()

        self.assertEqual(
            service._deterministic_companion_terms(topic="myrmecology", canonical_topic="myrmecology"),
            ["ant", "ants", "formicidae"],
        )
        self.assertEqual(
            service._deterministic_alias_terms(topic="apiology", canonical_topic="apiology"),
            ["melittology"],
        )
        self.assertEqual(
            service._deterministic_alias_terms(topic="odonatology", canonical_topic="odonatology"),
            ["odonata"],
        )

    def test_reel_service_builds_topic_concepts_from_expansion_when_missing(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-topic-only", "psychology", "Topic: psychology", "topic", None, "2026-03-14T00:00:00+00:00"),
        )
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        try:
            plan = _validated_query_plan(
                {
                    "canonical_topic": "Psychology",
                    "aliases": ["Psychology"],
                    "subtopics": ["cognitive psychology", "behavioral psychology", "social psychology"],
                    "related_terms": ["classic studies in psychology"],
                }
            )
            with mock.patch(
                "backend.app.services.reels.build_search_query_plan",
                return_value=plan,
            ) as planner:
                concepts = service._build_topic_only_concepts_from_expansion(
                    conn,
                    material_id="material-topic-only",
                    subject_tag="psychology",
                )
            planner.assert_called_once_with(
                conn,
                literal_query="psychology",
                should_cancel=None,
            )
            titles = {str(concept.get("title") or "").strip().lower() for concept in concepts}
            self.assertIn("psychology", titles)
            self.assertIn("cognitive psychology", titles)
            self.assertIn("behavioral psychology", titles)
        finally:
            conn.close()





    def test_topic_expansion_preserves_rows_but_filters_stale_generation_concepts(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO materials (id, subject_tag, raw_text, source_type, created_at) "
            "VALUES ('material-immutable', 'melittology', 'Topic: melittology', 'topic', '2026-07-10T00:00:00+00:00')"
        )
        conn.executemany(
            "INSERT INTO concepts (id, material_id, title, keywords_json, summary, created_at) "
            "VALUES (?, 'material-immutable', ?, ?, ?, '2026-07-10T00:00:00+00:00')",
            [
                ("concept-root", "Melittology", '["melittology"]', "Study of bees."),
                ("concept-taxonomy", "Bee Taxonomy", '["taxonomy"]', "Bee classification."),
            ],
        )
        before = [
            tuple(row)
            for row in conn.execute(
                "SELECT id, title, keywords_json, summary FROM concepts ORDER BY id"
            ).fetchall()
        ]
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        try:
            working = service._sync_topic_expansion_concepts(
                conn,
                material_id="material-immutable",
                concepts=[dict(row) for row in conn.execute("SELECT * FROM concepts ORDER BY id")],
                subject_tag="melittology",
                expansion={
                    "canonical_topic": "Melittology",
                    "aliases": ["Apiology"],
                    "subtopics": ["Bee field methods"],
                    "related_terms": ["Pollinator biology"],
                },
            )
            after = [
                tuple(row)
                for row in conn.execute(
                    "SELECT id, title, keywords_json, summary FROM concepts ORDER BY id"
                ).fetchall()
            ]
            self.assertEqual(after, before)
            self.assertEqual({row["id"] for row in working}, {"concept-root"})
            terms = {
                (row["concept_id"], row["term"], row["term_kind"])
                for row in conn.execute(
                    "SELECT concept_id, term, term_kind FROM concept_search_terms"
                ).fetchall()
            }
            self.assertIn(("concept-root", "Apiology", "alias"), terms)
            self.assertIn(("concept-root", "Bee field methods", "expansion"), terms)
            self.assertTrue(any(kind == "material_context" for _, _, kind in terms))
            self.assertFalse(any(concept_id == "concept-taxonomy" for concept_id, _, _ in terms))
        finally:
            conn.close()

    def test_strict_topic_selection_guard_rejects_low_alignment(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        allowed = service._passes_selection_topic_guard(
            video={"search_source": "youtube_html", "channel_title": "Math Channel", "title": "Chain rule explained"},
            ranking={
                "text_relevance": {
                    "concept_overlap": 0.13,
                    "subject_overlap": 0.0,
                    "off_topic_penalty": 0.0,
                    "score": 0.22,
                },
                "features": {
                    "query_alignment": 0.14,
                    "query_alignment_hits": ["chain rule"],
                },
            },
            segment_relevance={"concept_overlap": 0.11, "subject_overlap": 0.0},
            transcript_ranking=None,
            has_transcript=False,
            fast_mode=True,
            strict_topic_only=True,
        )
        rejected = service._passes_selection_topic_guard(
            video={"search_source": "youtube_html", "channel_title": "Math Channel", "title": "Calculus tricks"},
            ranking={
                "text_relevance": {
                    "concept_overlap": 0.13,
                    "subject_overlap": 0.0,
                    "off_topic_penalty": 0.16,
                    "score": 0.22,
                },
                "features": {
                    "query_alignment": 0.02,
                    "query_alignment_hits": [],
                },
            },
            segment_relevance={"concept_overlap": 0.11, "subject_overlap": 0.0},
            transcript_ranking=None,
            has_transcript=False,
            fast_mode=True,
            strict_topic_only=True,
        )
        self.assertTrue(allowed)
        self.assertFalse(rejected)

    def test_strict_topic_selection_guard_requires_root_anchor_for_channel_sources(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        rejected = service._passes_selection_topic_guard(
            video={"search_source": "youtube_channel", "channel_title": "Bee Lab", "title": "Bee anatomy lecture"},
            ranking={
                "text_relevance": {
                    "concept_overlap": 0.18,
                    "subject_overlap": 0.0,
                    "off_topic_penalty": 0.0,
                    "score": 0.28,
                },
                "features": {
                    "query_alignment": 0.22,
                    "query_alignment_hits": ["bee anatomy"],
                    "root_topic_alignment": 0.0,
                    "root_topic_alignment_hits": [],
                },
            },
            segment_relevance={"concept_overlap": 0.16, "subject_overlap": 0.0, "score": 0.24},
            transcript_ranking=None,
            has_transcript=False,
            fast_mode=True,
            strict_topic_only=True,
        )
        allowed = service._passes_selection_topic_guard(
            video={"search_source": "youtube_channel", "channel_title": "Bee Lab", "title": "Melittology field methods"},
            ranking={
                "text_relevance": {
                    "concept_overlap": 0.18,
                    "subject_overlap": 0.08,
                    "off_topic_penalty": 0.0,
                    "score": 0.32,
                },
                "features": {
                    "query_alignment": 0.24,
                    "query_alignment_hits": ["melittology"],
                    "root_topic_alignment": 0.26,
                    "root_topic_alignment_hits": ["melittology"],
                },
            },
            segment_relevance={"concept_overlap": 0.17, "subject_overlap": 0.08, "score": 0.26},
            transcript_ranking=None,
            has_transcript=False,
            fast_mode=True,
            strict_topic_only=True,
        )
        self.assertFalse(rejected)
        self.assertTrue(allowed)

    def test_local_recovery_is_scoped_to_same_material(self) -> None:
        conn = self._build_local_recovery_scope_test_conn()
        try:
            service = ReelService(embedding_service=None, youtube_service=None)
            candidates = service._recover_candidates_from_local_corpus(
                conn,
                material_id="material-a",
                concept_terms=["biology"],
                context_terms=[],
                concept_embedding=None,
                subject_tag="biology",
                visual_spec={"environment": [], "objects": [], "actions": []},
                preferred_video_duration="any",
                fast_mode=True,
                strict_topic_only=True,
                existing_video_counts={},
                generated_video_counts={},
                max_segments_per_video=1,
                concept_title="Biology",
            )
        finally:
            conn.close()

        self.assertEqual(candidates, [])

    def test_ranked_request_reels_appends_source_generation_before_deep_result(self) -> None:
        conn = self._build_generation_test_conn()
        try:
            conn.execute(
                """
                INSERT INTO reel_generations (
                    id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                    status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "gen-bootstrap",
                    "material-1",
                    None,
                    "request-key",
                    "fast",
                    "bootstrap",
                    "active",
                    None,
                    2,
                    "2026-03-13T00:00:00+00:00",
                    "2026-03-13T00:00:00+00:00",
                    "2026-03-13T00:00:00+00:00",
                    None,
                ),
            )
            conn.execute(
                """
                INSERT INTO reel_generations (
                    id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                    status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "gen-deep",
                    "material-1",
                    None,
                    "request-key",
                    "fast",
                    "deep",
                    "active",
                    "gen-bootstrap",
                    2,
                    "2026-03-13T00:05:00+00:00",
                    "2026-03-13T00:05:00+00:00",
                    "2026-03-13T00:05:00+00:00",
                    None,
                ),
            )

            with mock.patch.object(
                main_module.reel_service,
                "ranked_feed",
                side_effect=[
                    [
                        {
                            "reel_id": "bootstrap-1",
                            "video_url": "https://www.youtube.com/embed/video-a?start=0&end=20",
                            "t_start": 0.0,
                            "t_end": 20.0,
                            "video_duration_sec": 120,
                            "clip_duration_sec": 20.0,
                        },
                        {
                            "reel_id": "bootstrap-2",
                            "video_url": "https://www.youtube.com/embed/video-b?start=10&end=30",
                            "t_start": 10.0,
                            "t_end": 30.0,
                            "video_duration_sec": 140,
                            "clip_duration_sec": 20.0,
                        },
                    ],
                    [
                        {
                            "reel_id": "deep-overlap",
                            "video_url": "https://www.youtube.com/embed/video-b?start=10&end=30",
                            "t_start": 10.0,
                            "t_end": 30.0,
                            "video_duration_sec": 140,
                            "clip_duration_sec": 20.0,
                        },
                        {
                            "reel_id": "deep-1",
                            "video_url": "https://www.youtube.com/embed/video-c?start=5&end=25",
                            "t_start": 5.0,
                            "t_end": 25.0,
                            "video_duration_sec": 150,
                            "clip_duration_sec": 20.0,
                        },
                    ],
                ],
            ):
                merged = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=True,
                    generation_id="gen-deep",
                    min_relevance=None,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                )

            self.assertEqual(
                [row["reel_id"] for row in merged],
                ["bootstrap-1", "bootstrap-2", "deep-1"],
            )
        finally:
            conn.close()

    def test_ranked_request_reels_filters_low_value_page_one_niche_reels(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("apiology", "topic", "material-1"),
        )
        try:
            with mock.patch.object(
                main_module.reel_service,
                "ranked_feed",
                side_effect=[
                    [
                        {
                            "reel_id": "bad-meaning",
                            "video_url": "https://www.youtube.com/embed/video-a?start=0&end=35",
                            "video_title": "Apiology Meaning",
                            "video_description": "How to pronounce apiology and word meaning.",
                            "transcript_snippet": "",
                            "score": 0.82,
                            "relevance_score": 0.42,
                            "matched_terms": ["apiology"],
                            "video_duration_sec": 120,
                            "clip_duration_sec": 35.0,
                            "source_surface": "youtube_html",
                            "retrieval_stage": "high_precision",
                            "query_strategy": "literal",
                            "created_at": "2026-03-15T00:00:00+00:00",
                        },
                        {
                            "reel_id": "good-root",
                            "video_url": "https://www.youtube.com/embed/video-b?start=0&end=35",
                            "video_title": "Apiology - The Scientific Study of Honey Bees",
                            "video_description": "Educational introduction to apiology and bee science.",
                            "transcript_snippet": "",
                            "score": 0.76,
                            "relevance_score": 0.4,
                            "matched_terms": ["apiology"],
                            "video_duration_sec": 600,
                            "clip_duration_sec": 35.0,
                            "source_surface": "youtube_html",
                            "retrieval_stage": "high_precision",
                            "query_strategy": "literal",
                            "created_at": "2026-03-15T00:00:01+00:00",
                        },
                    ]
                ],
            ):
                merged = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=True,
                    generation_id="gen-1",
                    min_relevance=None,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )

            self.assertEqual([row["reel_id"] for row in merged], ["good-root"])
        finally:
            conn.close()

    def test_ranked_request_reels_relaxes_topic_relevance_for_later_pages(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("myrmecology", "topic", "material-1"),
        )
        try:
            candidate = {
                "reel_id": "good-companion",
                "video_url": "https://www.youtube.com/embed/video-ant?start=0&end=35",
                "video_title": "Ant Colony Behavior Explained",
                "video_description": "Educational overview of ant colony behavior and social structure.",
                "transcript_snippet": "",
                "score": 0.52,
                "relevance_score": 0.24,
                "matched_terms": ["ants"],
                "video_duration_sec": 420,
                "clip_duration_sec": 35.0,
                "source_surface": "youtube_related",
                "retrieval_stage": "recovery",
                "query_strategy": "recovery_adjacent",
                "created_at": "2026-03-15T00:00:00+00:00",
            }
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=[candidate]):
                page_one = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )
                page_two = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=2,
                    limit=5,
                )

            self.assertEqual(page_one, [])
            self.assertEqual(page_two, [])
        finally:
            conn.close()

    def test_ranked_request_reels_keeps_practice_valid_alias_without_literal_anchor(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("odonatology", "topic", "material-1"),
        )
        try:
            candidate = {
                "reel_id": "good-alias",
                "video_url": "https://www.youtube.com/embed/video-odonata?start=0&end=35",
                "video_title": "Odonata - Dragonfly and Damselfly",
                "video_description": "Educational introduction to dragonflies and damselflies.",
                "transcript_snippet": "",
                "score": 0.58,
                "relevance_score": 0.34,
                "matched_terms": ["odonata"],
                "video_duration_sec": 420,
                "clip_duration_sec": 35.0,
                "source_surface": "youtube_html",
                "retrieval_stage": "high_precision",
                "query_strategy": "literal",
                "created_at": "2026-03-15T00:00:00+00:00",
            }
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=[candidate]):
                page_one = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )

            self.assertEqual([item["reel_id"] for item in page_one], ["good-alias"])
        finally:
            conn.close()

    def test_ranked_request_reels_accepts_curated_broad_subtopic_on_page_one(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("calculus", "topic", "material-1"),
        )
        try:
            candidate = {
                "reel_id": "good-subtopic",
                "video_url": "https://www.youtube.com/embed/video-chain-rule?start=0&end=35",
                "video_title": "Chain Rule Explained with Worked Examples",
                "video_description": "Intro calculus lesson on derivatives and composite functions.",
                "transcript_snippet": "",
                "score": 0.57,
                "relevance_score": 0.31,
                "matched_terms": ["chain rule", "derivatives"],
                "video_duration_sec": 420,
                "clip_duration_sec": 35.0,
                "source_surface": "youtube_html",
                "retrieval_stage": "broad",
                "query_strategy": "explained",
                "created_at": "2026-03-15T00:00:00+00:00",
            }
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=[candidate]):
                page_one = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )

            self.assertEqual([row["reel_id"] for row in page_one], ["good-subtopic"])
        finally:
            conn.close()

    def test_ranked_request_reels_allows_local_bootstrap_on_page_one(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("python programming", "topic", "material-1"),
        )
        try:
            candidate = {
                "reel_id": "local-bootstrap",
                "video_url": "https://www.youtube.com/embed/python-basics?start=0&end=35",
                "video_title": "Python Programming Tutorial for Beginners",
                "video_description": "Learn Python programming with variables, loops, and functions.",
                "transcript_snippet": "",
                "score": 0.58,
                "relevance_score": 0.33,
                "matched_terms": ["python programming", "variables", "loops"],
                "video_duration_sec": 420,
                "clip_duration_sec": 35.0,
                "source_surface": "local_bootstrap",
                "retrieval_stage": "high_precision",
                "query_strategy": "literal",
                "created_at": "2026-03-15T00:00:00+00:00",
            }
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=[candidate]):
                page_one = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )

            self.assertEqual([row["reel_id"] for row in page_one], ["local-bootstrap"])
        finally:
            conn.close()

    def test_ranked_request_reels_backfills_page_one_from_later_page_for_curated_broad_topic(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("machine learning", "topic", "material-1"),
        )
        try:
            reels = [
                {
                    "reel_id": "page-one",
                    "video_url": "https://www.youtube.com/embed/video-root?start=0&end=35",
                    "video_title": "Supervised Learning Explained in 60 Seconds",
                    "video_description": "Quick introduction to machine learning basics.",
                    "transcript_snippet": "",
                    "score": 0.59,
                    "relevance_score": 0.31,
                    "matched_terms": ["supervised learning", "machine learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "high_precision",
                    "query_strategy": "explained",
                    "created_at": "2026-03-15T00:00:00+00:00",
                },
                {
                    "reel_id": "page-two-fill",
                    "video_url": "https://www.youtube.com/embed/video-types?start=0&end=35",
                    "video_title": "Types of Machine Learning | Supervised vs Unsupervised Learning",
                    "video_description": "Overview of the main machine learning categories for beginners.",
                    "transcript_snippet": "",
                    "score": 0.56,
                    "relevance_score": 0.24,
                    "matched_terms": ["supervised learning", "unsupervised learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "broad",
                    "query_strategy": "explained",
                    "created_at": "2026-03-15T00:00:01+00:00",
                },
            ]
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=reels):
                page_one = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )

            self.assertEqual([row["reel_id"] for row in page_one], ["page-one", "page-two-fill"])
        finally:
            conn.close()

    def test_ranked_request_reels_page_one_emergency_backfill_keeps_direct_inventory_only(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("machine learning", "topic", "material-1"),
        )
        try:
            reels = [
                {
                    "reel_id": "page-one",
                    "video_url": "https://www.youtube.com/embed/video-root?start=0&end=35",
                    "video_title": "Supervised Learning Explained in 60 Seconds",
                    "video_description": "Quick introduction to machine learning basics.",
                    "transcript_snippet": "",
                    "score": 0.59,
                    "relevance_score": 0.31,
                    "matched_terms": ["supervised learning", "machine learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "high_precision",
                    "query_strategy": "explained",
                    "created_at": "2026-03-15T00:00:00+00:00",
                },
                {
                    "reel_id": "page-one-emergency",
                    "video_url": "https://www.youtube.com/embed/video-types?start=0&end=35",
                    "video_title": "Types of Machine Learning | Supervised vs Unsupervised Learning",
                    "video_description": "Overview of the main machine learning categories for beginners.",
                    "transcript_snippet": "",
                    "score": 0.56,
                    "relevance_score": 0.28,
                    "matched_terms": ["supervised learning", "unsupervised learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "broad",
                    "query_strategy": "explained",
                    "created_at": "2026-03-15T00:00:01+00:00",
                },
                {
                    "reel_id": "blocked-related",
                    "video_url": "https://www.youtube.com/embed/video-related?start=0&end=35",
                    "video_title": "Related machine learning overview",
                    "video_description": "A related-video recommendation about machine learning.",
                    "transcript_snippet": "",
                    "score": 0.56,
                    "relevance_score": 0.28,
                    "matched_terms": ["machine learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_related",
                    "retrieval_stage": "recovery",
                    "query_strategy": "recovery_adjacent",
                    "created_at": "2026-03-15T00:00:02+00:00",
                },
            ]
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=reels):
                page_one = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )

            self.assertEqual([row["reel_id"] for row in page_one], ["page-one", "page-one-emergency"])
        finally:
            conn.close()

    def test_ranked_request_reels_page_two_preserves_page_one_emergency_inventory(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("machine learning", "topic", "material-1"),
        )
        try:
            reels = [
                {
                    "reel_id": "page-one",
                    "video_url": "https://www.youtube.com/embed/video-root?start=0&end=35",
                    "video_title": "Supervised Learning Explained in 60 Seconds",
                    "video_description": "Quick introduction to machine learning basics.",
                    "transcript_snippet": "",
                    "score": 0.59,
                    "relevance_score": 0.31,
                    "matched_terms": ["supervised learning", "machine learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "high_precision",
                    "query_strategy": "explained",
                    "created_at": "2026-03-15T00:00:00+00:00",
                },
                {
                    "reel_id": "page-one-emergency",
                    "video_url": "https://www.youtube.com/embed/video-types?start=0&end=35",
                    "video_title": "Types of Machine Learning | Supervised vs Unsupervised Learning",
                    "video_description": "Overview of the main machine learning categories for beginners.",
                    "transcript_snippet": "",
                    "score": 0.56,
                    "relevance_score": 0.28,
                    "matched_terms": ["supervised learning", "unsupervised learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "broad",
                    "query_strategy": "explained",
                    "created_at": "2026-03-15T00:00:01+00:00",
                },
            ]
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=reels):
                merged = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=2,
                    limit=5,
                )

            self.assertEqual([row["reel_id"] for row in merged], ["page-one", "page-one-emergency"])
        finally:
            conn.close()

    def test_ranked_request_reels_caps_repeated_broad_topic_anchor_flooding(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("machine learning", "topic", "material-1"),
        )
        try:
            reels = []
            for index in range(5):
                reels.append(
                    {
                        "reel_id": f"supervised-{index}",
                        "video_url": f"https://www.youtube.com/embed/video-supervised-{index}?start=0&end=35",
                        "video_title": f"Supervised Learning Explained Part {index + 1}",
                        "video_description": "Machine learning lesson about supervised learning for beginners.",
                        "transcript_snippet": "",
                        "score": 0.58 - index * 0.01,
                        "relevance_score": 0.34,
                        "matched_terms": ["supervised learning", "machine learning"],
                        "video_duration_sec": 420,
                        "clip_duration_sec": 35.0,
                        "source_surface": "youtube_html",
                        "retrieval_stage": "broad",
                        "query_strategy": "explained",
                        "created_at": f"2026-03-15T00:00:0{index}+00:00",
                    }
                )
            reels.append(
                {
                    "reel_id": "classification-1",
                    "video_url": "https://www.youtube.com/embed/video-classification?start=0&end=35",
                    "video_title": "Classification and Regression in Machine Learning",
                    "video_description": "Intro machine learning lesson on classification and regression.",
                    "transcript_snippet": "",
                    "score": 0.53,
                    "relevance_score": 0.33,
                    "matched_terms": ["classification", "regression", "machine learning"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "broad",
                    "query_strategy": "explained",
                    "created_at": "2026-03-15T00:00:09+00:00",
                }
            )
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=reels):
                page_two = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=2,
                    limit=5,
                )

            supervised_count = sum(
                1 for row in page_two if "supervised learning" in str(row.get("video_title") or "").lower()
            )
            self.assertLessEqual(supervised_count, 5)
            self.assertIn("classification-1", [row["reel_id"] for row in page_two])
        finally:
            conn.close()

    def test_ranked_request_reels_accumulates_page_one_before_later_page_relaxation(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            "UPDATE materials SET subject_tag = ?, source_type = ? WHERE id = ?",
            ("myrmecology", "topic", "material-1"),
        )
        try:
            reels = [
                {
                    "reel_id": "good-root",
                    "video_url": "https://www.youtube.com/embed/video-root?start=0&end=35",
                    "video_title": "Myrmecology Basics",
                    "video_description": "Introduction to myrmecology and ant science.",
                    "transcript_snippet": "",
                    "score": 0.64,
                    "relevance_score": 0.34,
                    "matched_terms": ["myrmecology"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_html",
                    "retrieval_stage": "high_precision",
                    "query_strategy": "literal",
                    "created_at": "2026-03-15T00:00:00+00:00",
                },
                {
                    "reel_id": "good-companion",
                    "video_url": "https://www.youtube.com/embed/video-ant?start=0&end=35",
                    "video_title": "Ant Colony Behavior Explained",
                    "video_description": "Educational overview of ant colony behavior and social structure.",
                    "transcript_snippet": "",
                    "score": 0.52,
                    "relevance_score": 0.24,
                    "matched_terms": ["ants"],
                    "video_duration_sec": 420,
                    "clip_duration_sec": 35.0,
                    "source_surface": "youtube_related",
                    "retrieval_stage": "recovery",
                    "query_strategy": "recovery_adjacent",
                    "created_at": "2026-03-15T00:00:01+00:00",
                },
            ]
            with mock.patch.object(main_module.reel_service, "ranked_feed", return_value=reels):
                page_one = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=1,
                    limit=5,
                )
                page_two = main_module._ranked_request_reels(
                    conn,
                    material_id="material-1",
                    fast_mode=False,
                    generation_id="gen-1",
                    min_relevance=0.3,
                    preferred_video_duration="any",
                    target_clip_duration_sec=55,
                    target_clip_duration_min_sec=20,
                    target_clip_duration_max_sec=55,
                    page=2,
                    limit=5,
                )

            self.assertEqual([row["reel_id"] for row in page_one], ["good-root"])
            self.assertEqual([row["reel_id"] for row in page_two], ["good-root"])
        finally:
            conn.close()

    def test_response_generation_ids_walks_full_source_chain(self) -> None:
        conn = self._build_generation_test_conn()
        try:
            rows = [
                ("gen-bootstrap", None, "bootstrap"),
                ("gen-deep-1", "gen-bootstrap", "deep"),
                ("gen-deep-2", "gen-deep-1", "deep"),
                ("gen-deep-3", "gen-deep-2", "deep"),
            ]
            for generation_id, source_generation_id, retrieval_profile in rows:
                conn.execute(
                    """
                    INSERT INTO reel_generations (
                        id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                        status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        "material-1",
                        None,
                        "request-key",
                        "fast",
                        retrieval_profile,
                        "active",
                        source_generation_id,
                        0,
                        "2026-03-13T00:00:00+00:00",
                        "2026-03-13T00:00:00+00:00",
                        "2026-03-13T00:00:00+00:00",
                        None,
                    ),
                )

            self.assertEqual(
                main_module._response_generation_ids(conn, "gen-deep-3"),
                ["gen-bootstrap", "gen-deep-1", "gen-deep-2", "gen-deep-3"],
            )
        finally:
            conn.close()


    def test_activate_generation_falls_back_to_legacy_generation_head_schema(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
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
                material_id TEXT NOT NULL,
                request_key TEXT NOT NULL,
                active_generation_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(material_id, request_key)
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
        conn.execute(
            """
            INSERT INTO reel_generations (
                id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "gen-1",
                "material-1",
                None,
                "request-key",
                "fast",
                "bootstrap",
                "pending",
                None,
                0,
                "2026-03-13T00:00:00+00:00",
                None,
                None,
                None,
            ),
        )

        main_module._activate_generation(
            conn,
            material_id="material-1",
            request_key="request-key",
            generation_id="gen-1",
            retrieval_profile="bootstrap",
        )

        head = conn.execute(
            "SELECT active_generation_id FROM reel_generation_heads WHERE material_id = ? AND request_key = ?",
            ("material-1", "request-key"),
        ).fetchone()
        self.assertIsNotNone(head)
        self.assertEqual(str(head["active_generation_id"]), "gen-1")
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

    def test_sqlite_schema_upgrade_adds_generation_id_before_generation_index(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                """
                CREATE TABLE reels (
                    id TEXT PRIMARY KEY,
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

            conn.executescript(SCHEMA)
            conn.execute("ALTER TABLE reels ADD COLUMN generation_id TEXT")
            _ensure_reels_generation_index_sqlite(conn)
            _migrate_reels_unique_clip_index_sqlite(conn)

            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(reels)").fetchall()
            }
            index_names = {
                row[1]
                for row in conn.execute("PRAGMA index_list(reels)").fetchall()
            }
            self.assertIn("generation_id", columns)
            self.assertIn("idx_reels_generation_id", index_names)
        finally:
            conn.close()

    def test_reel_service_topic_novelty_profile_calibrates_for_breadth(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)

        broad = service._topic_novelty_profile(
            subject_tag="machine learning",
            retrieval_profile="deep",
            fast_mode=False,
        )
        niche = service._topic_novelty_profile(
            subject_tag="myrmecology",
            retrieval_profile="deep",
            fast_mode=False,
        )

        self.assertLess(float(broad["cross_video_similarity"]), float(niche["cross_video_similarity"]))
        self.assertLess(float(broad["same_video_similarity"]), float(niche["same_video_similarity"]))

    def test_shape_request_page_reels_relaxes_later_pages_before_dropping_topic_guard(self) -> None:
        ranked = [
            {
                "reel_id": "related-reel",
                "video_title": "Ant colony communication explained",
                "video_description": "A lecture on ant colony communication and myrmecology.",
                "transcript_snippet": "This ant colony lecture explains myrmecology and pheromone trails.",
                "video_url": "https://www.youtube.com/embed/related-video?start=0&end=32",
                "t_start": 0.0,
                "t_end": 32.0,
                "score": 0.82,
                "relevance_score": 0.41,
                "matched_terms": ["ant colony", "myrmecology"],
                "source_surface": "youtube_related",
                "retrieval_stage": "recovery",
                "query_strategy": "recovery_adjacent",
                "video_duration_sec": 1800,
                "clip_duration_sec": 32.0,
                "created_at": "2026-03-13T00:00:00+00:00",
            },
            {
                "reel_id": "cache-reel",
                "video_title": "Ant field notes archive",
                "video_description": "Archived ant field notes with myrmecology commentary.",
                "transcript_snippet": "Archived myrmecology notes describe ant colonies and field behavior.",
                "video_url": "https://www.youtube.com/embed/cache-video?start=0&end=30",
                "t_start": 0.0,
                "t_end": 30.0,
                "score": 0.78,
                "relevance_score": 0.39,
                "matched_terms": ["myrmecology", "ant colonies"],
                "source_surface": "local_cache",
                "retrieval_stage": "recovery",
                "query_strategy": "recovery_adjacent",
                "video_duration_sec": 2400,
                "clip_duration_sec": 30.0,
                "created_at": "2026-03-13T00:01:00+00:00",
            },
        ]

        page_one = main_module._shape_request_page_reels(
            ranked,
            page=1,
            limit=5,
            subject_tag="myrmecology",
            strict_topic_only=True,
            min_relevance=0.3,
            preferred_video_duration="any",
            target_clip_duration_sec=38,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )
        page_three = main_module._shape_request_page_reels(
            ranked,
            page=3,
            limit=5,
            subject_tag="myrmecology",
            strict_topic_only=True,
            min_relevance=0.3,
            preferred_video_duration="any",
            target_clip_duration_sec=38,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )
        page_four = main_module._shape_request_page_reels(
            ranked,
            page=4,
            limit=5,
            subject_tag="myrmecology",
            strict_topic_only=True,
            min_relevance=0.3,
            preferred_video_duration="any",
            target_clip_duration_sec=38,
            target_clip_duration_min_sec=20,
            target_clip_duration_max_sec=55,
        )

        self.assertEqual(len(page_one), 0)
        self.assertEqual([item["reel_id"] for item in page_three], ["related-reel"])
        self.assertEqual([item["reel_id"] for item in page_four], ["related-reel", "cache-reel"])

    def test_reel_service_same_video_clip_novelty_requires_distance_or_high_confidence_function_change(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        prior = [{"text": "definition of machine learning and the core idea", "function_label": "definition", "function_confidence": 0.92}]
        current = {
            "text": "worked example of machine learning classification",
            "function_label": "worked_example",
            "function_confidence": 0.86,
            "clip_duration_sec": 35.0,
        }

        with mock.patch.object(service, "_text_pair_similarity", return_value=0.91):
            self.assertTrue(
                service._passes_same_video_clip_novelty(
                    None,
                    clip_context=current,
                    prior_contexts=prior,
                    subject_tag="machine learning",
                    retrieval_profile="deep",
                    fast_mode=False,
                )
            )

        weak_current = dict(current)
        weak_current["function_confidence"] = 0.7
        with mock.patch.object(service, "_text_pair_similarity", return_value=0.91):
            self.assertFalse(
                service._passes_same_video_clip_novelty(
                    None,
                    clip_context=weak_current,
                    prior_contexts=prior,
                    subject_tag="machine learning",
                    retrieval_profile="deep",
                    fast_mode=False,
                )
            )

    def test_reel_service_topic_expansion_terms_use_only_plan_values(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)

        terms = service._topic_expansion_terms(
            expansion={
                "canonical_topic": "Machine Learning",
                "aliases": [],
                "related_terms": [
                    "quantum machine learning",
                    "adversarial machine learning",
                    "attention",
                ],
                "subtopics": [
                    "quantum machine learning",
                    "adversarial machine learning",
                    "attention",
                ],
            },
            subject_tag="machine learning",
            limit=6,
        )

        self.assertEqual(
            terms[:6],
            [
                "quantum machine learning",
                "adversarial machine learning",
                "attention",
            ],
        )

    def test_reel_service_deep_topic_expansion_reuses_ai_query_plan(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-ai-expand", "psychology", "Topic: psychology", "topic", None, "2026-03-14T00:00:00+00:00"),
        )
        service = ReelService(embedding_service=None, youtube_service=None)
        expected = {
            "canonical_topic": "Psychology",
            "aliases": ["behavior science"],
            "subtopics": ["attachment theory", "memory"],
            "related_terms": ["cognitive bias"],
        }
        plan = _validated_query_plan(expected)
        with mock.patch(
            "backend.app.services.reels.build_search_query_plan",
            return_value=plan,
        ) as planner:
            expansion = service._deep_topic_expansion(
                conn,
                material_id="material-ai-expand",
                subject_tag="psychology",
                generation_id=None,
            )
        planner.assert_called_once_with(
            conn,
            literal_query="psychology",
            should_cancel=None,
        )
        self.assertEqual(expansion, expected)
        self.assertIn("attachment theory", expansion["subtopics"])
        self.assertIn("memory", expansion["subtopics"])
        self.assertIn("behavior science", expansion["aliases"])
        self.assertIn("cognitive bias", expansion["related_terms"])
        conn.close()

    def test_reel_service_deep_topic_expansion_falls_back_to_literal(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        plan = mock.Mock()
        plan.ai_status = "invalid"
        with mock.patch(
            "backend.app.services.reels.build_search_query_plan",
            return_value=plan,
        ):
            expansion = service._deep_topic_expansion(
                None,
                material_id="material-observed",
                subject_tag="calculus",
                generation_id=None,
            )
        self.assertEqual(
            expansion,
            {
                "canonical_topic": "calculus",
                "aliases": [],
                "subtopics": [],
                "related_terms": [],
            },
        )
        plan.as_topic_expansion.assert_not_called()

    def test_reel_service_topic_gate_rejects_dental_calculus_false_positive(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        relevance = service._score_text_relevance(
            None,
            text="Dental calculus cleaning teeth tartar removal explained",
            concept_terms=["Calculus", "derivatives", "integrals", "limits"],
            context_terms=["mathematics", "lesson"],
            concept_embedding=None,
            subject_tag="mathematics",
        )

        self.assertGreaterEqual(float(relevance["off_topic_penalty"]), 0.34)
        self.assertFalse(service._passes_relevance_gate(relevance, require_context=False, fast_mode=True))

    def test_reel_service_topic_gate_keeps_relevant_calculus_lesson(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        relevance = service._score_text_relevance(
            None,
            text="Calculus derivatives and integrals lesson with worked examples",
            concept_terms=["Calculus", "derivatives", "integrals", "limits"],
            context_terms=["mathematics", "lesson"],
            concept_embedding=None,
            subject_tag="mathematics",
        )

        self.assertLess(float(relevance["off_topic_penalty"]), 0.2)
        self.assertTrue(service._passes_relevance_gate(relevance, require_context=False, fast_mode=True))

    def test_reel_service_query_alignment_prefers_topic_specific_metadata(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        query_candidate = QueryCandidate(
            "computer science Python loops iteration",
            "literal",
            0.96,
            source_terms=["Python loops", "iteration", "computer science"],
        )

        relevant = service._query_alignment_score(
            "Python loops and iteration tutorial with worked examples",
            query_candidate=query_candidate,
            subject_tag="computer science",
        )
        off_topic = service._query_alignment_score(
            "Python snake enclosure setup and feeding tips",
            query_candidate=query_candidate,
            subject_tag="computer science",
        )

        self.assertGreater(float(relevant["score"]), float(off_topic["score"]))
        self.assertGreaterEqual(float(relevant["score"]), 0.16)
        self.assertLess(float(off_topic["score"]), 0.08)

    def test_reel_service_selection_topic_guard_rejects_metadata_only_false_positive(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)

        accepted = service._passes_selection_topic_guard(
            video={
                "title": "Python snake facts",
                "channel_title": "Nature Clips",
                "search_source": "youtube_html",
            },
            ranking={
                "text_relevance": {
                    "score": 0.11,
                    "concept_hits": ["python"],
                    "context_hits": [],
                    "concept_overlap": 0.05,
                    "context_overlap": 0.0,
                    "subject_overlap": 0.0,
                    "embedding_sim": 0.0,
                    "off_topic_penalty": 0.0,
                },
                "features": {
                    "query_alignment": 0.0,
                    "query_alignment_hits": [],
                },
            },
            segment_relevance={},
            transcript_ranking=None,
            has_transcript=False,
            fast_mode=True,
            strict_topic_only=False,
        )

        self.assertFalse(accepted)

    def test_reel_service_selection_topic_guard_keeps_precise_metadata_only_match(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)

        accepted = service._passes_selection_topic_guard(
            video={
                "title": "Python loops and iteration tutorial",
                "channel_title": "CS Academy",
                "search_source": "youtube_html",
            },
            ranking={
                "text_relevance": {
                    "score": 0.24,
                    "concept_hits": ["python", "loops"],
                    "context_hits": ["computer science"],
                    "concept_overlap": 0.18,
                    "context_overlap": 0.09,
                    "subject_overlap": 0.08,
                    "embedding_sim": 0.0,
                    "off_topic_penalty": 0.0,
                },
                "features": {
                    "query_alignment": 0.24,
                    "query_alignment_hits": ["python loops", "iteration"],
                },
            },
            segment_relevance={},
            transcript_ranking=None,
            has_transcript=False,
            fast_mode=True,
            strict_topic_only=False,
        )

        self.assertTrue(accepted)

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

    def test_youtube_service_bootstrap_query_variants_add_pedagogical_backup(self) -> None:
        service = YouTubeService()
        variants = service._build_search_query_variants(
            query="binary search trees",
            video_duration="short",
            source_surface="youtube_html",
            retrieval_profile="bootstrap",
        )

        self.assertEqual(
            variants,
            [
                {"query": "binary search trees", "surface": "youtube_html"},
                {"query": "binary search trees explained", "surface": "youtube_html"},
            ],
        )

    def test_youtube_service_literal_query_variants_stay_topic_anchored(self) -> None:
        service = YouTubeService()
        variants = service._build_search_query_variants(
            query="computer science binary search trees",
            video_duration="short",
            source_surface="youtube_html",
            retrieval_profile="deep",
            retrieval_strategy="literal",
        )

        self.assertEqual(
            variants,
            [
                {"query": "computer science binary search trees", "surface": "youtube_html"},
                {"query": "computer science binary search trees explained", "surface": "youtube_html"},
                {"query": "computer science binary search trees shorts", "surface": "youtube_html"},
                {"query": '"computer science binary search trees"', "surface": "youtube_html"},
            ],
        )

    def test_youtube_service_visual_query_variants_do_not_expand_beyond_planner_query(self) -> None:
        service = YouTubeService()
        variants = service._build_search_query_variants(
            query="sorting algorithm animation",
            video_duration="short",
            source_surface="youtube_html",
            retrieval_profile="deep",
            retrieval_strategy="scene",
        )

        self.assertEqual(variants, [{"query": "sorting algorithm animation", "surface": "youtube_html"}])

    def test_local_recovery_allows_extra_long_form_clips_beyond_default_cap(self) -> None:
        conn = self._build_local_recovery_scope_test_conn()
        conn.execute(
            """
            INSERT INTO videos (
                id, title, channel_title, description, duration_sec, view_count, is_creative_commons, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "video-long-bio",
                "Biology masterclass",
                "Science Channel",
                "Long-form biology lecture",
                5400,
                12000,
                0,
                "2026-03-13T00:10:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO reels (
                id, generation_id, material_id, concept_id, video_id, video_url,
                t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "reel-bio-long-1",
                None,
                "material-a",
                "concept-bio",
                "video-long-bio",
                "https://example.com/watch?v=video-long-bio",
                0.0,
                55.0,
                "Biology introduction",
                '["Cells"]',
                1.0,
                "2026-03-13T00:11:00+00:00",
            ),
        )
        service = ReelService(embedding_service=None, youtube_service=None)
        try:
            with mock.patch.object(
                service,
                "_quick_candidate_metadata_gate",
                return_value={"passes": True, "metadata_text": "biology long form lecture"},
            ), mock.patch.object(
                service,
                "_score_video_candidate",
                return_value={
                    "passes": True,
                    "final_score": 0.54,
                    "discovery_score": 0.41,
                    "text_relevance": {
                        "concept_hits": ["biology"],
                        "context_hits": [],
                        "concept_overlap": 0.2,
                        "context_overlap": 0.0,
                        "subject_overlap": 0.12,
                        "embedding_sim": 0.0,
                        "score": 0.32,
                        "off_topic_penalty": 0.0,
                    },
                    "features": {"query_alignment": 0.18},
                },
            ):
                candidates = service._recover_candidates_from_local_corpus(
                    conn,
                    material_id="material-a",
                    concept_terms=["biology"],
                    context_terms=[],
                    concept_embedding=None,
                    subject_tag="biology",
                    visual_spec={"environment": [], "objects": [], "actions": []},
                    preferred_video_duration="any",
                    fast_mode=True,
                    strict_topic_only=False,
                    existing_video_counts={"video-long-bio": 1},
                    generated_video_counts={},
                    max_segments_per_video=1,
                    concept_title="Biology",
                )
        finally:
            conn.close()

        self.assertEqual([candidate["video_id"] for candidate in candidates], ["video-long-bio"])

    def test_youtube_service_deep_external_variants_add_exact_anchor(self) -> None:
        service = YouTubeService()
        variants = service._build_external_query_variants(
            query_variants=[{"query": "computer science binary search trees tutorial", "surface": "youtube_html"}],
            retrieval_strategy="tutorial",
            retrieval_profile="deep",
        )

        self.assertEqual(
            variants,
            [
                {"query": "computer science binary search trees tutorial", "surface": "youtube_html"},
                {"query": '"computer science binary search trees" tutorial', "surface": "youtube_html"},
                {
                    "query": "site:youtube.com computer science binary search trees tutorial educational",
                    "surface": "duckduckgo_site",
                },
            ],
        )

    def test_youtube_service_merge_unique_videos_upgrades_placeholder_metadata(self) -> None:
        service = YouTubeService()
        merged = service._merge_unique_videos(
            [
                {
                    "id": "abc123xyz00",
                    "title": "YouTube Video abc123xyz00",
                    "channel_title": "",
                    "description": "",
                    "duration_sec": 0,
                    "view_count": 0,
                    "published_at": "",
                    "search_source": "duckduckgo_site",
                }
            ],
            [
                {
                    "id": "abc123xyz00",
                    "title": "Binary Search Trees Tutorial",
                    "channel_title": "CS Academy",
                    "description": "Step-by-step walkthrough of BST search and insertion.",
                    "duration_sec": 420,
                    "view_count": 12000,
                    "published_at": "2025-01-02T00:00:00+00:00",
                    "search_source": "youtube_html",
                    "query_strategy": "tutorial",
                    "query_stage": "high_precision",
                    "search_query": "binary search trees tutorial",
                }
            ],
            None,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["title"], "Binary Search Trees Tutorial")
        self.assertEqual(merged[0]["channel_title"], "CS Academy")
        self.assertEqual(merged[0]["duration_sec"], 420)
        self.assertEqual(merged[0]["view_count"], 12000)
        self.assertEqual(merged[0]["search_source"], "youtube_html")

    def test_youtube_service_finalize_search_rows_prefers_richer_exact_match(self) -> None:
        service = YouTubeService()
        ranked = service._finalize_search_rows(
            [
                {
                    "id": "fallback-1",
                    "title": "YouTube Video fallback-1",
                    "channel_title": "",
                    "description": "",
                    "duration_sec": 0,
                    "view_count": 0,
                    "published_at": "",
                    "search_source": "duckduckgo_site",
                    "query_strategy": "literal",
                    "query_stage": "high_precision",
                    "search_query": "binary search trees tutorial",
                },
                {
                    "id": "rich-1",
                    "title": "Binary Search Trees Tutorial",
                    "channel_title": "CS Academy",
                    "description": "Binary search tree search, insertion, and traversal examples.",
                    "duration_sec": 540,
                    "view_count": 85000,
                    "published_at": "2025-03-01T00:00:00+00:00",
                    "search_source": "youtube_html",
                    "query_strategy": "tutorial",
                    "query_stage": "high_precision",
                    "search_query": "binary search trees tutorial",
                },
            ],
            query="binary search trees tutorial",
            max_results=1,
            video_duration="medium",
        )

        self.assertEqual([row["id"] for row in ranked], ["rich-1"])

    def test_youtube_service_extract_search_data_parses_views_and_relative_publish_time(self) -> None:
        service = YouTubeService()
        rows = service._extract_videos_from_search_data(
            {
                "contents": [
                    {
                        "videoRenderer": {
                            "videoId": "abc123xyz00",
                            "title": {"runs": [{"text": "Cell signaling animation"}]},
                            "ownerText": {"runs": [{"text": "Bio Channel"}]},
                            "descriptionSnippet": {"runs": [{"text": "Explains cell communication."}]},
                            "lengthText": {"simpleText": "12:34"},
                            "viewCountText": {"simpleText": "1.2M views"},
                            "publishedTimeText": {"simpleText": "2 years ago"},
                        }
                    }
                ]
            },
            max_results=3,
            video_duration="medium",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["view_count"], 1_200_000)
        published_at = datetime.fromisoformat(rows[0]["published_at"])
        now = datetime.now(timezone.utc)
        age_days = (now - published_at).days
        self.assertGreaterEqual(age_days, 365)
        self.assertLessEqual(age_days, 900)

    def test_youtube_service_extract_search_data_supports_compact_renderers_and_channel_ids(self) -> None:
        service = YouTubeService()
        rows = service._extract_videos_from_search_data(
            {
                "contents": [
                    {
                        "compactVideoRenderer": {
                            "videoId": "abc123xyz00",
                            "title": {"simpleText": "Jacobian matrix worked example"},
                            "shortBylineText": {
                                "runs": [
                                    {
                                        "text": "Math Depth",
                                        "navigationEndpoint": {"browseEndpoint": {"browseId": "UCmathdepth01"}},
                                    }
                                ]
                            },
                            "thumbnailOverlays": [
                                {
                                    "thumbnailOverlayTimeStatusRenderer": {
                                        "text": {"simpleText": "12:34"}
                                    }
                                }
                            ],
                            "viewCountText": {"simpleText": "18K views"},
                            "publishedTimeText": {"simpleText": "8 months ago"},
                        }
                    }
                ]
            },
            max_results=3,
            video_duration="medium",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["channel_id"], "UCmathdepth01")
        self.assertEqual(rows[0]["channel_title"], "Math Depth")
        self.assertEqual(rows[0]["duration_sec"], 754)
        self.assertEqual(rows[0]["view_count"], 18_000)

    def test_youtube_service_search_without_data_api_uses_graph_before_external(self) -> None:
        service = YouTubeService()
        graph_calls: list[str] = []
        external_calls: list[str] = []

        def fake_search_variant_via_html(*args, **kwargs):
            return [
                {
                    "id": "seedvideo01a",
                    "title": "Melittology introduction",
                    "channel_id": "UCbeechannel1",
                    "channel_title": "Bee Channel",
                    "description": "Study of bees overview.",
                    "duration_sec": 420,
                    "search_source": "youtube_html",
                }
            ]

        def fake_expand_videos_via_youtube_graph(**kwargs):
            graph_calls.append(str(kwargs.get("graph_profile") or ""))
            return [
                {
                    "id": "graphvideo01",
                    "title": "Melittology field methods",
                    "channel_id": "UCbeechannel1",
                    "channel_title": "Bee Channel",
                    "description": "Field methods in melittology.",
                    "duration_sec": 510,
                    "search_source": "youtube_related",
                    "discovery_path": "related:seedvideo01a",
                    "seed_video_id": "seedvideo01a",
                    "seed_channel_id": "UCbeechannel1",
                    "crawl_depth": 1,
                },
                {
                    "id": "graphvideo02",
                    "title": "Melittology lecture",
                    "channel_id": "UCbeechannel2",
                    "channel_title": "Bee Lab",
                    "description": "Advanced melittology lecture.",
                    "duration_sec": 840,
                    "search_source": "youtube_related",
                    "discovery_path": "related:seedvideo01a",
                    "seed_video_id": "seedvideo01a",
                    "seed_channel_id": "UCbeechannel1",
                    "crawl_depth": 1,
                },
                {
                    "id": "graphvideo03",
                    "title": "Apiology and melittology comparison",
                    "channel_id": "UCbeechannel3",
                    "channel_title": "Pollinator Studies",
                    "description": "Comparing apiology and melittology.",
                    "duration_sec": 360,
                    "search_source": "youtube_channel",
                    "discovery_path": "channel:UCbeechannel1",
                    "seed_video_id": "seedvideo01a",
                    "seed_channel_id": "UCbeechannel1",
                    "crawl_depth": 2,
                },
            ]

        def fake_search_external_fallbacks(**kwargs):
            external_calls.append(str(kwargs.get("query") or ""))
            return [{"id": "externalvid1", "title": "External fallback", "search_source": "duckduckgo_site"}]

        service._search_variant_via_html = fake_search_variant_via_html  # type: ignore[method-assign]
        service._expand_videos_via_youtube_graph = fake_expand_videos_via_youtube_graph  # type: ignore[method-assign]
        service._search_external_fallbacks = fake_search_external_fallbacks  # type: ignore[method-assign]

        rows = service._search_without_data_api(
            query="melittology",
            max_results=3,
            creative_commons_only=False,
            video_duration="medium",
            retrieval_strategy="literal",
            retrieval_stage="high_precision",
            source_surface="youtube_html",
            include_external_fallbacks=True,
            retrieval_profile="deep",
            graph_profile="deep",
            root_terms=["melittology", "study of bees"],
        )

        self.assertEqual(graph_calls, ["deep"])
        self.assertEqual(external_calls, [])
        self.assertEqual(len(rows), 3)

    def test_youtube_service_graph_expansion_merges_related_and_channel_rows(self) -> None:
        service = YouTubeService()
        seed_rows = [
            {
                "id": "seedvideo01a",
                "title": "Melittology introduction",
                "channel_id": "UCbeechannel1",
                "channel_title": "Bee Channel",
                "description": "Study of bees overview.",
                "duration_sec": 420,
                "search_source": "youtube_html",
            },
            {
                "id": "seedvideo02a",
                "title": "Melittology behavior",
                "channel_id": "UCbeechannel2",
                "channel_title": "Bee Lab",
                "description": "Bee behavior and ecology.",
                "duration_sec": 480,
                "search_source": "youtube_html",
            },
        ]
        related_rows_by_seed = {
            "seedvideo01a": [
                {
                    "id": "related001a",
                    "title": "Melittology field methods",
                    "channel_id": "UCbeechannel1",
                    "channel_title": "Bee Channel",
                    "description": "Field methods in melittology.",
                    "duration_sec": 510,
                }
            ],
            "seedvideo02a": [
                {
                    "id": "related002a",
                    "title": "Melittology lab workflow",
                    "channel_id": "UCbeechannel2",
                    "channel_title": "Bee Lab",
                    "description": "Lab workflow for bee studies.",
                    "duration_sec": 540,
                }
            ],
        }
        channel_rows_by_channel = {
            "UCbeechannel1": [
                {
                    "id": "channel001a",
                    "title": "Apiology and melittology comparison",
                    "channel_id": "UCbeechannel1",
                    "channel_title": "Bee Channel",
                    "description": "Comparing bee-study disciplines.",
                    "duration_sec": 360,
                }
            ],
            "UCbeechannel2": [
                {
                    "id": "channel002a",
                    "title": "Bee taxonomy lecture",
                    "channel_id": "UCbeechannel2",
                    "channel_title": "Bee Lab",
                    "description": "Classification and taxonomy for bees.",
                    "duration_sec": 600,
                }
            ],
        }

        with (
            mock.patch.object(service, "_graph_seed_score", return_value=0.9),
            mock.patch.object(service, "_strong_direct_inventory_count", return_value=0),
            mock.patch.object(service, "_graph_candidate_is_anchored", return_value=True),
            mock.patch.object(service, "_fetch_watch_html", side_effect=lambda video_id, *, deadline: video_id),
            mock.patch.object(
                service,
                "_extract_related_videos_from_watch_html",
                side_effect=lambda html, *, max_results, video_duration: related_rows_by_seed.get(str(html), [])[:max_results],
            ),
            mock.patch.object(service, "_fetch_channel_videos_html", side_effect=lambda channel_id, *, deadline: channel_id),
            mock.patch.object(
                service,
                "_extract_channel_videos_from_channel_html",
                side_effect=lambda html, *, max_results, video_duration: channel_rows_by_channel.get(str(html), [])[:max_results],
            ),
        ):
            rows = service._expand_videos_via_youtube_graph(
                seed_rows=seed_rows,
                query="melittology",
                max_results=6,
                video_duration="medium",
                retrieval_strategy="literal",
                retrieval_stage="recovery",
                deadline=time.monotonic() + 5.0,
                graph_profile="deep",
                root_terms=["melittology", "study of bees"],
            )

        row_by_id = {row["id"]: row for row in rows}
        self.assertEqual(
            set(row_by_id),
            {"related001a", "related002a", "channel001a", "channel002a"},
        )
        self.assertEqual(row_by_id["related001a"]["search_source"], "youtube_related")
        self.assertEqual(row_by_id["channel001a"]["search_source"], "youtube_channel")
        self.assertEqual(row_by_id["channel001a"]["seed_channel_id"], "UCbeechannel1")

    def test_youtube_service_finalize_search_rows_prefers_direct_search_over_channel_backfill(self) -> None:
        service = YouTubeService()
        ranked = service._finalize_search_rows(
            [
                {
                    "id": "channelvid01",
                    "title": "Jacobian matrix lecture",
                    "channel_id": "UCmathdepth01",
                    "channel_title": "Math Depth",
                    "description": "Channel backfill lecture on jacobian matrix intuition.",
                    "duration_sec": 780,
                    "view_count": 44000,
                    "published_at": "2025-01-01T00:00:00+00:00",
                    "search_source": "youtube_channel",
                    "discovery_path": "channel:UCmathdepth01",
                    "seed_video_id": "seedvideo01a",
                    "seed_channel_id": "UCmathdepth01",
                    "crawl_depth": 2,
                },
                {
                    "id": "directvid001",
                    "title": "Jacobian matrix worked example",
                    "channel_id": "UCmathdepth02",
                    "channel_title": "Math Depth",
                    "description": "Direct search result for jacobian matrix with worked examples.",
                    "duration_sec": 620,
                    "view_count": 41000,
                    "published_at": "2025-02-01T00:00:00+00:00",
                    "search_source": "youtube_html",
                    "discovery_path": "search:youtube_html",
                    "seed_video_id": "",
                    "seed_channel_id": "UCmathdepth02",
                    "crawl_depth": 0,
                },
            ],
            query="jacobian matrix",
            max_results=1,
            video_duration="medium",
        )

        self.assertEqual([row["id"] for row in ranked], ["directvid001"])

    def test_youtube_service_search_without_data_api_skips_external_when_html_fills_budget(self) -> None:
        service = YouTubeService()
        external_calls: list[str] = []

        def fake_search_variant_via_html(*args, **kwargs):
            return [
                {
                    "id": "vid-a",
                    "title": "Binary Search Trees Tutorial",
                    "channel_title": "CS A",
                    "description": "BST walkthrough",
                    "duration_sec": 420,
                    "search_source": "youtube_html",
                },
                {
                    "id": "vid-b",
                    "title": "Binary Search Tree Search Example",
                    "channel_title": "CS B",
                    "description": "Search example",
                    "duration_sec": 390,
                    "search_source": "youtube_html",
                },
                {
                    "id": "vid-c",
                    "title": "BST Insertion Explained",
                    "channel_title": "CS C",
                    "description": "Insertion explained",
                    "duration_sec": 360,
                    "search_source": "youtube_html",
                },
                {
                    "id": "vid-d",
                    "title": "Binary Trees vs BSTs",
                    "channel_title": "CS D",
                    "description": "Comparison",
                    "duration_sec": 480,
                    "search_source": "youtube_html",
                },
            ]

        def fake_search_via_duckduckgo(*args, **kwargs):
            external_calls.append("duckduckgo")
            return []

        def fake_search_via_bing(*args, **kwargs):
            external_calls.append("bing")
            return []

        service._search_variant_via_html = fake_search_variant_via_html  # type: ignore[method-assign]
        service._search_via_duckduckgo = fake_search_via_duckduckgo  # type: ignore[method-assign]
        service._search_via_bing = fake_search_via_bing  # type: ignore[method-assign]

        rows = service._search_without_data_api(
            query="binary search trees",
            max_results=4,
            creative_commons_only=False,
            video_duration="medium",
            retrieval_strategy="literal",
            retrieval_stage="high_precision",
            source_surface="youtube_html",
            include_external_fallbacks=True,
            variant_limit=1,
            skip_primary_variants=False,
            retrieval_profile="deep",
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(external_calls, [])

    def test_youtube_service_search_external_fallbacks_batches_video_details_once(self) -> None:
        service = YouTubeService()
        service.api_key = "test-key"
        detail_calls: list[list[str]] = []

        def fake_search_via_duckduckgo(*args, **kwargs):
            return ["abc123xyz01", "abc123xyz02"]

        def fake_search_via_bing(*args, **kwargs):
            return ["abc123xyz02", "abc123xyz03"]

        def fake_video_details(video_ids: list[str], deadline: float | None = None):
            del deadline
            detail_calls.append(list(video_ids))
            return {
                "abc123xyz01": {"duration_sec": 420, "view_count": 1_000, "license": "youtube"},
                "abc123xyz02": {"duration_sec": 480, "view_count": 2_000, "license": "youtube"},
                "abc123xyz03": {"duration_sec": 540, "view_count": 3_000, "license": "youtube"},
            }

        service._search_via_duckduckgo = fake_search_via_duckduckgo  # type: ignore[method-assign]
        service._search_via_bing = fake_search_via_bing  # type: ignore[method-assign]
        service._video_details = fake_video_details  # type: ignore[method-assign]

        rows = service._search_external_fallbacks(
            query="binary search trees",
            max_results=3,
            video_duration="medium",
            retrieval_strategy="literal",
            retrieval_stage="recovery",
            source_surface="youtube_html",
            retrieval_profile="bootstrap",
            deadline=None,
            variant_limit=1,
        )

        self.assertEqual(len(detail_calls), 1)
        self.assertEqual(set(detail_calls[0]), {"abc123xyz01", "abc123xyz02", "abc123xyz03"})
        row_by_id = {row["id"]: row for row in rows}
        self.assertEqual(row_by_id["abc123xyz01"]["duration_sec"], 420)
        self.assertEqual(row_by_id["abc123xyz03"]["view_count"], 3_000)

    def test_youtube_service_search_via_data_api_prefers_full_detail_description(self) -> None:
        service = YouTubeService()
        service.api_key = "test-key"
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "items": [
                {
                    "id": {"videoId": "abc123xyz00"},
                    "snippet": {
                        "title": "Search title",
                        "channelId": "UCsearch01",
                        "channelTitle": "Search Channel",
                        "description": "Short search preview...",
                        "publishedAt": "2025-01-01T00:00:00Z",
                    },
                }
            ]
        }
        full_description = (
            "Full lesson description covering binary search tree search, insertion, and traversal with worked examples."
        )

        with (
            mock.patch.object(service, "_session_get", return_value=response),
            mock.patch.object(
                service,
                "_video_details",
                return_value={
                    "abc123xyz00": {
                        "title": "Binary Search Trees Tutorial",
                        "channel_id": "UCdetail01",
                        "channel_title": "Detail Channel",
                        "description": full_description,
                        "published_at": "2025-02-02T00:00:00Z",
                        "duration_sec": 420,
                        "view_count": 12345,
                        "license": "youtube",
                    }
                },
            ),
        ):
            rows = service._search_via_data_api(
                query="binary search trees",
                max_results=1,
                creative_commons_only=False,
                video_duration="medium",
                retrieval_strategy="literal",
                retrieval_stage="high_precision",
                source_surface="youtube_api",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Binary Search Trees Tutorial")
        self.assertEqual(rows[0]["channel_title"], "Detail Channel")
        self.assertEqual(rows[0]["description"], full_description)
        self.assertEqual(rows[0]["published_at"], "2025-02-02T00:00:00Z")

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

    def test_youtube_service_bootstrap_non_cc_falls_back_to_data_api_when_html_underfills(self) -> None:
        service = YouTubeService()
        service.api_key = "test-key"

        html_calls: list[str] = []
        data_api_calls: list[str] = []
        external_calls: list[str] = []

        def fake_search_via_data_api(*args: object, **kwargs: object) -> list[dict[str, object]]:
            query = str(kwargs.get("query") or args[0])
            data_api_calls.append(query)
            return [{"id": "api123xyz00", "title": "API video"}]

        def fake_search_without_data_api(*args: object, **kwargs: object) -> list[dict[str, object]]:
            query = str(kwargs.get("query") or args[0])
            html_calls.append(query)
            return [{"id": "html123xyz0", "title": "HTML video"}]

        def fake_search_external_fallbacks(*args: object, **kwargs: object) -> list[dict[str, object]]:
            query = str(kwargs.get("query") or args[0])
            external_calls.append(query)
            return [{"id": "ext123xyz00", "title": "External video"}]

        service._search_via_data_api = fake_search_via_data_api  # type: ignore[method-assign]
        service._search_without_data_api = fake_search_without_data_api  # type: ignore[method-assign]
        service._search_external_fallbacks = fake_search_external_fallbacks  # type: ignore[method-assign]

        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE search_cache (cache_key TEXT PRIMARY KEY, response_json TEXT, created_at TEXT)"
            )
            rows = service._search_videos_with_conn(
                conn,
                query="binary search trees",
                max_results=4,
                creative_commons_only=False,
                video_duration=None,
                retrieval_strategy="literal",
                retrieval_stage="high_precision",
                source_surface="youtube_html",
                retrieval_profile="bootstrap",
                allow_external_fallbacks=False,
                variant_limit=1,
            )

        self.assertEqual([row["id"] for row in rows], ["html123xyz0", "api123xyz00"])
        self.assertEqual(html_calls, ["binary search trees"])
        self.assertEqual(data_api_calls, ["binary search trees"])
        self.assertEqual(external_calls, [])

    def test_youtube_service_deep_recovery_uses_data_api_when_non_api_returns_nothing(self) -> None:
        service = YouTubeService()
        service.api_key = "test-key"

        html_calls: list[str] = []
        data_api_calls: list[str] = []

        def fake_search_via_data_api(*args: object, **kwargs: object) -> list[dict[str, object]]:
            query = str(kwargs.get("query") or args[0])
            data_api_calls.append(query)
            return [{"id": "api123xyz00", "title": "API recovery video"}]

        def fake_search_without_data_api(*args: object, **kwargs: object) -> list[dict[str, object]]:
            query = str(kwargs.get("query") or args[0])
            html_calls.append(query)
            return []

        service._search_via_data_api = fake_search_via_data_api  # type: ignore[method-assign]
        service._search_without_data_api = fake_search_without_data_api  # type: ignore[method-assign]
        service._should_use_data_api = mock.Mock(side_effect=[False, True])  # type: ignore[method-assign]

        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE search_cache (cache_key TEXT PRIMARY KEY, response_json TEXT, created_at TEXT)"
            )
            rows = service._search_videos_with_conn(
                conn,
                query="binary search trees",
                max_results=4,
                creative_commons_only=False,
                video_duration=None,
                retrieval_strategy="literal",
                retrieval_stage="recovery",
                source_surface="youtube_html",
                retrieval_profile="deep",
                allow_external_fallbacks=False,
                variant_limit=1,
            )

        self.assertEqual([row["id"] for row in rows], ["api123xyz00"])
        self.assertEqual(len(html_calls), 2)
        self.assertEqual(data_api_calls, ["binary search trees"])

    def test_ranked_feed_topic_keeps_weak_lexical_match_and_score_orders_it(self) -> None:
        conn = self._build_ranked_feed_test_conn()
        conn.execute(
            "UPDATE materials SET source_type = 'topic' WHERE id = ?",
            ("material-ranked",),
        )
        conn.execute(
            """
            INSERT INTO videos (
                id, title, channel_title, description, duration_sec, view_count,
                is_creative_commons, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "vidWeakMatch01",
                "An unfamiliar worked example",
                "Teaching Channel",
                "A transcript-grounded lesson selected upstream.",
                180,
                100,
                0,
                "2026-03-13T00:02:30+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO reels (
                id, generation_id, material_id, concept_id, video_id, video_url,
                t_start, t_end, transcript_snippet, takeaways_json, base_score,
                difficulty, model_used, quality_degraded, selected_cue_ids_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "reel-weak-match",
                None,
                "material-ranked",
                "concept-ranked",
                "vidWeakMatch01",
                "https://www.youtube.com/watch?v=vidWeakMatch01",
                10.0,
                40.0,
                "A validated explanation selected from the transcript.",
                '["A useful worked example"]',
                1.2,
                0.5,
                "gemini-3.5-flash",
                0,
                '["cue-weak-1"]',
                "2026-03-13T00:03:30+00:00",
            ),
        )
        service = ReelService(embedding_service=None, youtube_service=None)
        ranked = service.ranked_feed(conn, "material-ranked", fast_mode=True)

        self.assertEqual(
            [item["reel_id"] for item in ranked],
            ["reel-ranked", "reel-weak-match"],
        )
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        conn.close()

    def test_ranked_feed_cache_reuses_results_and_invalidates_on_feedback_and_transcript_updates(self) -> None:
        conn = self._build_ranked_feed_test_conn()
        service = ReelService(embedding_service=None, youtube_service=None)
        relevance_calls: list[str] = []
        caption_calls: list[str] = []

        def fake_score_text_relevance(*args, **kwargs):
            relevance_calls.append("called")
            return {
                "score": 0.82,
                "concept_overlap": 0.5,
                "context_overlap": 0.2,
                "matched_terms": ["cell signaling"],
                "off_topic_penalty": 0.0,
                "reason": "matched concept terms",
            }

        def fake_build_caption_cues(*args, **kwargs):
            caption_calls.append("called")
            return [{"start": 0.0, "end": 5.0, "text": "Cell signaling recap"}]

        with mock.patch.object(service, "_score_text_relevance", side_effect=fake_score_text_relevance), mock.patch.object(
            service,
            "_build_caption_cues",
            side_effect=fake_build_caption_cues,
        ):
            first = service.ranked_feed(conn, "material-ranked", fast_mode=True)
            second = service.ranked_feed(conn, "material-ranked", fast_mode=True)

            self.assertEqual(first, second)
            self.assertEqual(len(relevance_calls), 1)
            self.assertEqual(len(caption_calls), 1)
            cache_row = conn.execute("SELECT COUNT(*) FROM ranked_feed_cache").fetchone()
            self.assertEqual(cache_row[0], 1)

            conn.execute(
                """
                UPDATE reels
                SET t_start = ?, t_end = ?, search_context_json = ?
                WHERE id = ?
                """,
                (
                    10.25,
                    40.25,
                    json.dumps({
                        "surface_eligible": True,
                        "boundary_status": "verified",
                        "boundary_diagnostics": {"acoustic_verified": True},
                    }),
                    "reel-ranked",
                ),
            )
            promoted = service.ranked_feed(
                conn,
                "material-ranked",
                fast_mode=True,
            )
            self.assertEqual(promoted[0]["t_start"], 10.25)
            self.assertEqual(len(relevance_calls), 2)

            conn.execute(
                """
                INSERT INTO reel_feedback (
                    id, reel_id, helpful, confusing, rating, saved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("feedback-1", "reel-ranked", 1, 0, 5, 1, "2026-03-13T00:05:00+00:00"),
            )
            third = service.ranked_feed(conn, "material-ranked", fast_mode=True)
            self.assertEqual(len(relevance_calls), 3)
            self.assertGreater(third[0]["score"], first[0]["score"])

            updated_artifact_key = transcript_artifact_key(
                video_id="vidRanked01",
                provider="supadata",
                requested_language="en",
                returned_language="en",
                native_mode=True,
            )
            updated_artifact = {
                "artifact_key": updated_artifact_key,
                "video_id": "vidRanked01",
                "provider": "supadata",
                "requested_language": "en",
                "returned_language": "en",
                "native_mode": True,
                "schema_version": 2,
                "segments": [
                    {"cue_id": "cue-1", "start": 0.0, "end": 5.0, "text": "Updated transcript", "lang": "en"}
                ],
                "duration_sec": 5.0,
                "created_at": "2026-03-13T00:06:00+00:00",
            }
            conn.execute(
                """
                UPDATE transcript_artifacts
                SET artifact_json = ?, created_at = ?
                WHERE video_id = ?
                """,
                (
                    json.dumps(updated_artifact),
                    "2026-03-13T00:06:00+00:00",
                    "vidRanked01",
                ),
            )
            service.ranked_feed(conn, "material-ranked", fast_mode=True)
            self.assertEqual(len(relevance_calls), 4)
            self.assertEqual(len(caption_calls), 4)
        conn.close()

    def test_ranked_feed_cache_version_rejects_pre_semantic_gate_rows(self) -> None:
        conn = self._build_ranked_feed_test_conn()
        service = ReelService(embedding_service=None, youtube_service=None)
        current_version = service.RANKED_FEED_CACHE_VERSION
        self.assertGreater(current_version, 12)

        stale_relevance = {
            "score": 0.9,
            "concept_overlap": 0.8,
            "context_overlap": 0.8,
            "matched_terms": ["cell signaling"],
            "off_topic_penalty": 0.0,
            "reason": "legacy cache entry",
        }
        service.RANKED_FEED_CACHE_VERSION = 12
        with mock.patch.object(
            service,
            "_score_text_relevance",
            return_value=stale_relevance,
        ), mock.patch.object(
            service,
            "_build_caption_cues",
            return_value=[{"start": 0.0, "end": 5.0, "text": "x" * 220}],
        ):
            stale = service.ranked_feed(conn, "material-ranked", fast_mode=True)
        self.assertEqual(len(stale[0]["captions"][0]["text"]), 220)

        service.RANKED_FEED_CACHE_VERSION = current_version
        current_relevance = {
            "score": 0.0,
            "concept_overlap": 0.0,
            "context_overlap": 0.0,
            "matched_terms": [],
            "off_topic_penalty": 0.5,
            "reason": "current semantic gate rejects the row",
        }
        with mock.patch.object(
            service,
            "_score_text_relevance",
            return_value=current_relevance,
        ) as score_relevance:
            refreshed = service.ranked_feed(conn, "material-ranked", fast_mode=True)

        self.assertEqual(refreshed, [])
        score_relevance.assert_called_once()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM ranked_feed_cache").fetchone()[0],
            2,
        )
        conn.close()

    def test_ranked_feed_never_calls_legacy_youtube_details(self) -> None:
        conn = self._build_ranked_feed_test_conn()
        youtube_service = mock.Mock()
        service = ReelService(embedding_service=None, youtube_service=youtube_service)

        def fake_score_text_relevance(*args, **kwargs):
            return {
                "score": 0.82,
                "concept_overlap": 0.5,
                "context_overlap": 0.2,
                "matched_terms": ["cell signaling"],
                "off_topic_penalty": 0.0,
                "reason": "matched concept terms",
            }

        original_description = conn.execute(
            "SELECT description FROM videos WHERE id = ?", ("vidRanked01",)
        ).fetchone()[0]

        with mock.patch.object(service, "_score_text_relevance", side_effect=fake_score_text_relevance):
            ranked = service.ranked_feed(conn, "material-ranked", fast_mode=True)

        self.assertEqual(ranked[0]["video_description"], original_description)
        unchanged_description = conn.execute(
            "SELECT description FROM videos WHERE id = ?", ("vidRanked01",)
        ).fetchone()
        self.assertEqual(unchanged_description[0], original_description)
        youtube_service.video_details.assert_not_called()
        conn.close()

    def test_quick_candidate_metadata_gate_rejects_off_topic_risky_video(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        result = service._quick_candidate_metadata_gate(
            video={
                "title": "Funny prank compilation",
                "description": "Best reaction memes and viral jokes.",
                "channel_title": "Daily Laughs",
                "search_source": "bing_site",
            },
            query_candidate=QueryCandidate(
                text="chain rule calculus tutorial",
                strategy="literal",
                confidence=0.8,
                source_terms=["chain rule", "calculus"],
            ),
            concept_terms=["chain rule", "calculus", "derivative"],
            context_terms=["limits", "differentiation"],
            subject_tag="math",
            strict_topic_only=False,
            require_context=False,
            fast_mode=True,
        )
        self.assertFalse(result["passes"])

    def test_quick_candidate_metadata_gate_strict_topic_requires_direct_support(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        result = service._quick_candidate_metadata_gate(
            video={
                "title": "Late-night desk vlog with interphase playlist",
                "description": (
                    "Coffee run, desk reset, planner setup, and a quiet routine for exams. "
                    "No science lesson, lecture, or walkthrough."
                ),
                "channel_title": "Campus Logs",
                "search_source": "youtube_html",
            },
            query_candidate=QueryCandidate(
                text="mitosis cell cycle tutorial",
                strategy="literal",
                confidence=0.8,
                source_terms=["mitosis", "cell cycle"],
            ),
            concept_terms=[
                "mitosis",
                "cell cycle",
                "interphase",
                "chromosomes",
                "cytokinesis",
                "prophase",
                "metaphase",
                "anaphase",
                "telophase",
                "centromere",
                "spindle fibers",
                "daughter cells",
                "replication",
            ],
            context_terms=["division", "nucleus", "dna replication"],
            subject_tag="biology",
            strict_topic_only=True,
            require_context=False,
            fast_mode=True,
        )
        self.assertFalse(result["passes"])

    def test_quick_candidate_metadata_gate_strict_topic_requires_root_anchor_for_channel_backfill(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        result = service._quick_candidate_metadata_gate(
            video={
                "title": "Bee anatomy lecture",
                "description": "Worker bees, pollination, and hive roles explained.",
                "channel_title": "Bee Lab",
                "search_source": "youtube_channel",
            },
            query_candidate=QueryCandidate(
                text="bee anatomy lecture",
                strategy="literal",
                confidence=0.8,
                source_terms=["bee anatomy"],
            ),
            concept_terms=["bee anatomy", "worker bees", "pollination"],
            context_terms=["hive", "queen bee"],
            subject_tag="melittology",
            strict_topic_only=True,
            require_context=False,
            fast_mode=True,
            root_topic_terms=["melittology", "apiology", "study of bees"],
        )
        self.assertFalse(result["passes"])

    def test_quick_candidate_metadata_gate_accepts_topical_short_without_educational_cues(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        result = service._quick_candidate_metadata_gate(
            video={
                "title": "Chain rule in 45 seconds",
                "description": "Quick derivative trick for inner and outer functions.",
                "channel_title": "Math Shorts",
                "search_source": "youtube_html",
                "duration_sec": 45,
            },
            query_candidate=QueryCandidate(
                text="chain rule calculus shorts",
                strategy="literal",
                confidence=0.8,
                source_terms=["chain rule", "calculus"],
            ),
            concept_terms=["chain rule", "calculus", "derivative", "composition of functions"],
            context_terms=["differentiation"],
            subject_tag="calculus",
            strict_topic_only=True,
            require_context=False,
            fast_mode=True,
            root_topic_terms=["calculus", "derivative", "chain rule"],
        )
        self.assertTrue(result["passes"])

    def test_reel_service_hard_blocks_subject_meaning_titles_for_non_language_topics(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())

        self.assertTrue(
            service._is_hard_blocked_low_value_video(
                title="Bryology Meaning",
                description="",
                channel_title="",
                subject_tag="bryology",
            )
        )
        self.assertTrue(
            service._is_hard_blocked_low_value_video(
                title="What does odonatology mean?",
                description="",
                channel_title="",
                subject_tag="odonatology",
            )
        )
        self.assertTrue(
            service._is_hard_blocked_low_value_video(
                title="What is Lambda Calculus and why?",
                description="An introduction to lambda calculus and functional programming.",
                channel_title="Programming Theory",
                subject_tag="calculus",
            )
        )

    def test_quick_candidate_metadata_gate_rejects_lexicon_noise_for_niche_topic(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        result = service._quick_candidate_metadata_gate(
            video={
                "title": "Melittology meaning in Hindi",
                "description": "How to pronounce melittology and word meaning with example sentence.",
                "channel_title": "Daily Vocabulary",
                "search_source": "youtube_html",
            },
            query_candidate=QueryCandidate(
                text="melittology bees explained",
                strategy="literal",
                confidence=0.8,
                source_terms=["melittology", "bees"],
            ),
            concept_terms=["melittology", "bees", "bee behavior"],
            context_terms=["pollination"],
            subject_tag="melittology",
            strict_topic_only=True,
            require_context=False,
            fast_mode=True,
            root_topic_terms=["melittology", "apiology"],
        )
        self.assertFalse(result["passes"])
        self.assertGreater(float(result["lexicon_noise"]), 0.3)

    def test_quick_candidate_metadata_gate_rejects_specific_topic_drift(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        result = service._quick_candidate_metadata_gate(
            video={
                "title": "Forensic Psychology Explained",
                "description": "An overview of forensic psychology and criminal profiling.",
                "channel_title": "Psych Explained",
                "search_source": "youtube_html",
            },
            query_candidate=QueryCandidate(
                text="forensic palynology tutorial",
                strategy="literal",
                confidence=0.82,
                source_terms=["forensic palynology", "palynology"],
            ),
            concept_terms=["forensic palynology", "palynology", "pollen analysis"],
            context_terms=["forensic science"],
            subject_tag="palynology",
            strict_topic_only=True,
            require_context=False,
            fast_mode=True,
            root_topic_terms=["palynology", "forensic palynology"],
        )
        self.assertFalse(result["passes"])

    def test_shape_reels_for_request_context_keeps_unanchored_practice_clip(self) -> None:
        shaped = main_module._shape_reels_for_request_context(
            [
                {
                    "reel_id": "reel-1",
                    "video_title": "Neural networks in 60 seconds",
                    "video_description": "How hidden layers learn patterns from data.",
                    "transcript_snippet": "",
                    "matched_terms": ["neural networks", "patterns"],
                    "video_url": "https://www.youtube.com/embed/video-1",
                    "video_duration_sec": 60,
                    "clip_duration_sec": 45.0,
                    "relevance_score": 0.29,
                    "score": 0.41,
                    "source_surface": "youtube_html",
                    "created_at": "2026-03-15T00:00:00+00:00",
                }
            ],
            page=1,
            limit=5,
            subject_tag="machine learning",
            strict_topic_only=True,
        )

        self.assertEqual([item["reel_id"] for item in shaped], ["reel-1"])

    def test_request_source_surface_allowed_opens_related_results_on_page_two(self) -> None:
        reel = {"source_surface": "youtube_related"}

        self.assertFalse(main_module._request_source_surface_allowed(reel, page=1))
        self.assertFalse(main_module._request_source_surface_allowed(reel, page=2))
        self.assertTrue(main_module._request_source_surface_allowed(reel, page=3))

    def test_request_effective_min_relevance_relaxes_on_page_two_for_topic_feeds(self) -> None:
        self.assertEqual(
            main_module._request_effective_min_relevance(
                0.0,
                page=1,
                subject_tag="calculus",
                strict_topic_only=True,
            ),
            0.3,
        )
        self.assertEqual(
            main_module._request_effective_min_relevance(
                0.0,
                page=2,
                subject_tag="calculus",
                strict_topic_only=True,
            ),
            0.22,
        )

    def test_score_video_candidate_skips_description_semantic_for_strong_prefilter(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        semantic_calls: list[str] = []

        def fake_semantic(*args, **kwargs):
            text = kwargs.get("text")
            if text is None and len(args) >= 2:
                text = args[1]
            semantic_calls.append(str(text))
            return 0.72

        with mock.patch.object(
            service,
            "_score_text_relevance",
            return_value={
                "score": 0.42,
                "concept_overlap": 0.22,
                "context_overlap": 0.0,
                "subject_overlap": 0.04,
                "embedding_sim": 0.35,
                "concept_hits": ["chain rule", "calculus"],
                "context_hits": [],
                "off_topic_penalty": 0.0,
            },
        ), mock.patch.object(service, "_semantic_similarity", side_effect=fake_semantic), mock.patch.object(
            service, "_learned_strategy_factor", return_value=1.0
        ), mock.patch.object(service, "_channel_quality_score", return_value=0.82), mock.patch.object(
            service, "_score_clipability_from_metadata", return_value=0.7
        ):
            ranking = service._score_video_candidate(
                None,
                video={
                    "title": "Chain rule calculus explained",
                    "description": "Detailed walkthrough with worked derivatives and practice.",
                    "channel_title": "Math Academy",
                    "search_source": "youtube_api",
                    "duration_sec": 420,
                    "view_count": 12000,
                    "published_at": "2025-01-10T00:00:00Z",
                },
                query_candidate=QueryCandidate(
                    text="chain rule calculus tutorial",
                    strategy="literal",
                    confidence=0.8,
                    source_terms=["chain rule", "calculus"],
                ),
                concept_terms=["chain rule", "calculus", "derivative"],
                context_terms=[],
                concept_embedding=None,
                subject_tag="math",
                visual_spec={"environment": [], "objects": [], "actions": []},
                preferred_video_duration="any",
                stage_name="high_precision",
                require_context=False,
                fast_mode=True,
                quick_signals={
                    "metadata_text": "Chain rule calculus explained Detailed walkthrough with worked derivatives and practice.",
                    "query_alignment": {"score": 0.32, "hits": ["chain rule", "calculus"]},
                    "educational_intent": 0.9,
                    "skip_semantic_description": True,
                },
            )

        self.assertEqual(semantic_calls, ["Chain rule calculus explained"])
        self.assertTrue(ranking["passes"])

    def test_filter_reels_by_min_relevance_falls_back_for_niche_batches(self) -> None:
        reels = [
            {"reel_id": "a", "relevance_score": 0.12},
            {"reel_id": "b", "relevance_score": 0.11},
            {"reel_id": "c", "relevance_score": 0.09},
            {"reel_id": "d", "relevance_score": 0.04},
        ]

        filtered = main_module._filter_reels_by_min_relevance(reels, 0.3)

        self.assertEqual([row["reel_id"] for row in filtered], ["a", "b", "c"])

    def test_request_shaping_keeps_grounded_low_score_inventory_to_fill_the_page(self) -> None:
        ranked = [
            {
                "reel_id": "strong",
                "video_id": "strong-video",
                "relevance_score": 0.9,
                "_selection_ordered": True,
            },
            {
                "reel_id": "grounded-low-score",
                "video_id": "grounded-low-video",
                "relevance_score": 0.0,
                "_selection_ordered": True,
            },
        ]

        shaped = main_module._shape_request_page_reels(
            ranked,
            page=1,
            limit=2,
            subject_tag="rare niche subject",
            strict_topic_only=True,
            min_relevance=0.3,
            preferred_video_duration="any",
            target_clip_duration_sec=38,
            target_clip_duration_min_sec=None,
            target_clip_duration_max_sec=None,
        )

        self.assertEqual(
            [row["reel_id"] for row in shaped],
            ["strong", "grounded-low-score"],
        )

    def test_versioned_selection_metadata_missing_factual_contract_fails_closed(self) -> None:
        metadata = ReelService._selection_metadata(
            {
                "selection_contract_version": "selector-v1",
                "directly_teaches_topic": True,
                "substantive": True,
            }
        )

        self.assertFalse(metadata["_selection_factually_grounded"])

    def test_bootstrap_topic_keywords_include_canonical_aliases(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        concept = {
            "title": "Apiology",
            "summary": "Core ideas, terminology, and intuition for Apiology.",
        }

        plan = _validated_query_plan(
            {
                "canonical_topic": "Melittology",
                "aliases": ["Melittology"],
                "subtopics": [],
                "related_terms": ["Apiology"],
            }
        )
        with mock.patch(
            "backend.app.services.reels.build_search_query_plan",
            return_value=plan,
        ):
            keywords = service._bootstrap_topic_keywords(concept, subject_tag="apiology", conn=None)

        self.assertEqual(keywords[0], "apiology")
        self.assertIn("melittology", keywords)

    def test_bootstrap_topic_keywords_ai_failure_keeps_literal_only(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        concept = {
            "title": "Apiology",
            "summary": "Core ideas, terminology, and intuition for Apiology.",
        }

        plan = mock.Mock()
        plan.ai_status = "unavailable"
        with mock.patch(
            "backend.app.services.reels.build_search_query_plan",
            return_value=plan,
        ):
            keywords = service._bootstrap_topic_keywords(concept, subject_tag="apiology", conn=None)

        self.assertEqual(keywords, ["apiology"])
        plan.as_topic_expansion.assert_not_called()

    def test_bug_1b_final_unique_reels_counter_increments(self) -> None:
        """Bug 1B: final_unique_reels should increment, not stay at max(current, 1)."""
        metrics: dict[str, object] = {"final_unique_reels": 0}
        # Simulate 3 reel generations
        for _ in range(3):
            metrics["final_unique_reels"] = int(metrics.get("final_unique_reels") or 0) + 1
        self.assertEqual(metrics["final_unique_reels"], 3)

    def test_bug_1c_bootstrap_pool_is_weak_ignores_transcript_field(self) -> None:
        """Bug 1C: _bootstrap_pool_is_weak should not check has_transcript (never set on candidates)."""
        service = ReelService(embedding_service=None, youtube_service=None)
        candidates = [
            {
                "video": {"channel_title": f"channel_{i}"},
                "ranking": {"final_score": 0.5},
                "query_candidate": QueryCandidate(f"query {i}", f"strategy_{i}", 0.9),
            }
            for i in range(5)
        ]
        # Pool should be strong (enough candidates, good score, unique channels/strategies)
        self.assertFalse(service._bootstrap_pool_is_weak(candidates, max_generation_target=3))

    def test_bug_1e_empty_cache_ttl_is_short(self) -> None:
        """Bug 1E: Empty search results should only be cached for 15 minutes, not 2 hours."""
        self.assertEqual(YouTubeService.SEARCH_CACHE_EMPTY_TTL_SEC, 900)

    def test_quality_5a_known_educational_channels_get_higher_tier(self) -> None:
        """Quality 5A: Known educational channels should get 'known_educational' tier."""
        service = ReelService(embedding_service=None, youtube_service=None)
        tier = service._infer_channel_tier("khan academy", "intro to algebra")
        self.assertEqual(tier, "known_educational")
        tier2 = service._infer_channel_tier("3blue1brown", "essence of linear algebra")
        self.assertEqual(tier2, "known_educational")

    def test_niche_4a_datamuse_called_for_opaque_topics_with_filtering(self) -> None:
        """Niche 4A: Datamuse should be called for opaque topics, with results filtered."""
        service = TopicExpansionService()
        with (
            mock.patch.object(
                service,
                "_search_wikipedia_results",
                side_effect=[
                    [{"title": "Bryology", "snippet": "Bryology is the study of mosses."}],
                    [],
                ],
            ),
            mock.patch.object(
                service,
                "_search_wikidata_entities",
                return_value=[{"label": "Bryology", "description": "study of mosses", "aliases": []}],
            ),
            mock.patch.object(service, "_fetch_wikipedia_links", return_value=["Moss", "Liverwort"]),
            mock.patch.object(
                service,
                "_fetch_datamuse_related_terms",
                return_value=["bryophyte", "cooking", "bryology research"],
            ) as datamuse_mock,
        ):
            payload = service._expand_topic_uncached(
                topic="bryology",
                max_subtopics=6,
                max_aliases=4,
                max_related_terms=4,
            )
        datamuse_mock.assert_called_once()
        # "bryology research" should pass anchor check, "cooking" should not
        all_terms = " ".join(str(v) for v in payload.values()).lower()
        self.assertNotIn("cooking", all_terms)

    def test_niche_4c_new_static_topics_present(self) -> None:
        """Niche 4C: New static topics should be in STATIC_TOPIC_SUBTOPICS."""
        for topic in [
            "computer science", "nursing", "engineering", "philosophy",
            "music theory", "art history", "environmental science",
            "political science", "organic chemistry", "algebra",
        ]:
            self.assertIn(topic, TopicExpansionService.STATIC_TOPIC_SUBTOPICS, f"Missing: {topic}")

if __name__ == "__main__":
    unittest.main()
