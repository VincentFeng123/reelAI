import json
import sqlite3
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.app.main as main_module
from backend.app.main import _resolve_target_clip_duration_bounds
from backend.app.db import SCHEMA, _ensure_reels_generation_index_sqlite, _migrate_reels_unique_clip_index_sqlite
from backend.app.services.material_intelligence import MaterialIntelligenceService
from backend.app.services.reels import ConceptIntentPlan, QueryCandidate, ReelService, TranscriptPrefetchTask
from backend.app.services.topic_expansion import TopicExpansionService
from backend.app.services.youtube import YouTubeService
from backend.app.main import _build_generation_request_key
from fastapi.testclient import TestClient


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
                "video-ranked",
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
                "video-ranked",
                "https://example.com/watch?v=video-ranked",
                0.0,
                30.0,
                "Cell signaling recap",
                '["Signal reception"]',
                1.2,
                "2026-03-13T00:03:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO transcript_cache (
                video_id, transcript_json, created_at
            ) VALUES (?, ?, ?)
            """,
            (
                "video-ranked",
                '[{"start": 0.0, "duration": 5.0, "text": "Cell signaling recap"}]',
                "2026-03-13T00:04:00+00:00",
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

    def test_material_intelligence_seeds_topic_only_material_with_subtopics(self) -> None:
        service = MaterialIntelligenceService()
        service.client = None
        concepts, objectives = service.extract_concepts_and_objectives(
            None,
            "Topic: calculus",
            subject_tag="calculus",
            max_concepts=6,
        )
        titles = {str(concept.get("title") or "").strip().lower() for concept in concepts}
        self.assertIn("calculus", titles)
        self.assertTrue({"limits", "derivatives", "chain rule"}.intersection(titles))
        self.assertTrue(any("calculus" in objective.lower() for objective in objectives))

    def test_material_intelligence_uses_topic_expansion_for_unmapped_topic(self) -> None:
        service = MaterialIntelligenceService()
        service.client = None
        with mock.patch.object(
            service.topic_expansion_service,
            "expand_topic",
            return_value={
                "canonical_topic": "Spanish language",
                "aliases": ["Spanish language"],
                "subtopics": ["grammar", "verb conjugation", "pronunciation", "common phrases"],
                "related_terms": ["conversation practice", "listening comprehension"],
            },
        ):
            concepts, objectives = service.extract_concepts_and_objectives(
                None,
                "Topic: spanish",
                subject_tag="spanish",
                max_concepts=6,
            )
        titles = {str(concept.get("title") or "").strip().lower() for concept in concepts}
        self.assertIn("spanish", titles)
        self.assertTrue({"grammar", "verb conjugation", "pronunciation"}.intersection(titles))
        root_concept = next(
            concept for concept in concepts if str(concept.get("title") or "").strip().lower() == "spanish"
        )
        root_keywords = {str(keyword).strip().lower() for keyword in (root_concept.get("keywords") or [])}
        self.assertIn("spanish language", root_keywords)
        self.assertIn("grammar", root_keywords)
        self.assertTrue(any("spanish" in objective.lower() for objective in objectives))

    def test_material_intelligence_psychology_topic_has_real_fallback_subtopics(self) -> None:
        service = MaterialIntelligenceService()
        service.client = None
        with mock.patch.object(service.topic_expansion_service, "_request_json", return_value={}):
            concepts, objectives = service.extract_concepts_and_objectives(
                None,
                "Topic: psychology",
                subject_tag="psychology",
                max_concepts=6,
            )
        titles = {str(concept.get("title") or "").strip().lower() for concept in concepts}
        self.assertIn("psychology", titles)
        self.assertTrue(
            {
                "cognitive psychology",
                "behavioral psychology",
                "developmental psychology",
                "social psychology",
            }.intersection(titles)
        )
        self.assertNotIn("topic", titles)
        self.assertNotIn("topic psychology", titles)
        self.assertTrue(any("psychology" in objective.lower() for objective in objectives))

    def test_material_intelligence_topic_seed_precedes_llm_for_topic_only_material(self) -> None:
        service = MaterialIntelligenceService()
        service.client = object()
        with (
            mock.patch.object(
                service.topic_expansion_service,
                "expand_topic",
                return_value={
                    "canonical_topic": "Spanish language",
                    "aliases": ["Spanish language"],
                    "subtopics": ["grammar", "verb conjugation", "pronunciation"],
                    "related_terms": ["conversation practice"],
                },
            ),
            mock.patch.object(
                service,
                "_cached_or_generate",
                return_value={
                    "concepts": [
                        {"title": f"Generic {index}", "keywords": ["generic"], "summary": "generic summary"}
                        for index in range(8)
                    ],
                    "objectives": ["generic objective"],
                },
            ),
        ):
            concepts, objectives = service.extract_concepts_and_objectives(
                None,
                "Topic: spanish",
                subject_tag="spanish",
                max_concepts=4,
            )
        titles = [str(concept.get("title") or "").strip().lower() for concept in concepts]
        self.assertEqual(titles[0], "spanish")
        self.assertIn("grammar", titles)
        self.assertNotEqual(objectives[0].lower(), "generic objective")

    def test_material_intelligence_opaque_topic_seed_does_not_promote_companion_terms_to_concepts(self) -> None:
        service = MaterialIntelligenceService()
        service.client = None
        with mock.patch.object(
            service.topic_expansion_service,
            "expand_topic",
            return_value={
                "canonical_topic": "Melittology",
                "aliases": [],
                "subtopics": ["Melittology field methods"],
                "related_terms": ["bees"],
            },
        ):
            concepts, _objectives = service.extract_concepts_and_objectives(
                None,
                "Topic: melittology",
                subject_tag="melittology",
                max_concepts=4,
            )
        titles = {str(concept.get("title") or "").strip().lower() for concept in concepts}
        self.assertIn("melittology", titles)
        self.assertIn("melittology field methods", titles)
        self.assertNotIn("bees", titles)

    def test_material_intelligence_opaque_topic_only_material_skips_llm_garbage(self) -> None:
        service = MaterialIntelligenceService()
        service.client = object()
        with (
            mock.patch.object(
                service.topic_expansion_service,
                "expand_topic",
                return_value={
                    "canonical_topic": "Melittology",
                    "aliases": [],
                    "subtopics": ["Melittology field methods"],
                    "related_terms": ["bees"],
                },
            ),
            mock.patch.object(
                service,
                "_cached_or_generate",
                return_value={
                    "concepts": [
                        {"title": "Bees", "keywords": ["bees"], "summary": "General bees"},
                        {"title": "Human-animal Interaction", "keywords": ["interaction"], "summary": "Off topic"},
                    ],
                    "objectives": ["generic objective"],
                },
            ) as cached_mock,
        ):
            concepts, objectives = service.extract_concepts_and_objectives(
                None,
                "Topic: apiology",
                subject_tag="apiology",
                max_concepts=4,
            )
        titles = {str(concept.get("title") or "").strip().lower() for concept in concepts}
        cached_mock.assert_not_called()
        self.assertIn("apiology", titles)
        self.assertNotIn("bees", titles)
        self.assertNotIn("human-animal interaction", titles)
        self.assertTrue(any("apiology" in objective.lower() for objective in objectives))

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
        datamuse_mock.assert_not_called()
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
        datamuse_mock.assert_not_called()
        self.assertIn("ants", related_terms)
        self.assertIn("ants", search_terms)
        self.assertNotIn("ants mythology", related_terms)
        self.assertNotIn("fungi myology", related_terms)

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
            with mock.patch.object(
                service.topic_expansion_service,
                "expand_topic",
                return_value={
                    "canonical_topic": "Psychology",
                    "aliases": ["Psychology"],
                    "subtopics": ["cognitive psychology", "behavioral psychology", "social psychology"],
                    "related_terms": ["classic studies in psychology"],
                },
            ):
                concepts = service._build_topic_only_concepts_from_expansion(
                    conn,
                    material_id="material-topic-only",
                    subject_tag="psychology",
                )
            titles = {str(concept.get("title") or "").strip().lower() for concept in concepts}
            self.assertIn("psychology", titles)
            self.assertIn("cognitive psychology", titles)
            self.assertIn("behavioral psychology", titles)
        finally:
            conn.close()

    def test_reel_service_sync_topic_expansion_rewrites_generic_topic_concepts(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-rewrite", "psychology", "Topic: psychology", "topic", None, "2026-03-14T00:00:00+00:00"),
        )
        concept_rows = [
            ("concept-root", "Psychology", '["psychology"]', "Psychology overview"),
            ("concept-a", "Psychology Foundations", '["psychology"]', "Foundations"),
            ("concept-b", "Topic", '["topic"]', "Topic"),
            ("concept-c", "Psychology Worked Examples", '["psychology"]', "Worked examples"),
        ]
        for concept_id, title, keywords_json, summary in concept_rows:
            conn.execute(
                """
                INSERT INTO concepts (
                    id, material_id, title, keywords_json, summary, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept_id,
                    "material-rewrite",
                    title,
                    keywords_json,
                    summary,
                    None,
                    "2026-03-14T00:00:00+00:00",
                ),
            )
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        try:
            updated = service._sync_topic_expansion_concepts(
                conn,
                material_id="material-rewrite",
                concepts=conn.execute(
                    "SELECT id, material_id, title, keywords_json, summary, embedding_json, created_at FROM concepts WHERE material_id = ? ORDER BY created_at ASC, id ASC",
                    ("material-rewrite",),
                ).fetchall(),
                subject_tag="psychology",
                expansion={
                    "canonical_topic": "Psychology",
                    "aliases": ["Psychology"],
                    "subtopics": ["cognitive psychology", "behavioral psychology", "social psychology"],
                    "related_terms": ["classic studies in psychology"],
                },
            )
            titles = {str(concept.get("title") or "").strip().lower() for concept in updated}
            self.assertIn("cognitive psychology", titles)
            self.assertIn("behavioral psychology", titles)
            self.assertIn("social psychology", titles)
            self.assertNotIn("topic", titles)
            self.assertNotIn("psychology foundations", titles)
        finally:
            conn.close()

    def test_reel_service_sync_topic_expansion_filters_non_anchored_niche_concepts(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-melittology", "melittology", "Topic: melittology", "topic", None, "2026-03-14T00:00:00+00:00"),
        )
        concept_rows = [
            ("concept-a", "Bee Taxonomy", '["bee taxonomy"]', "Taxonomy of bees"),
            ("concept-b", "Entomology", '["entomology"]', "General insect science"),
            ("concept-c", "Pollinator Biology", '["pollinator biology"]', "General pollinator overview"),
        ]
        for concept_id, title, keywords_json, summary in concept_rows:
            conn.execute(
                """
                INSERT INTO concepts (
                    id, material_id, title, keywords_json, summary, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept_id,
                    "material-melittology",
                    title,
                    keywords_json,
                    summary,
                    None,
                    "2026-03-14T00:00:00+00:00",
                ),
            )
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        try:
            updated = service._sync_topic_expansion_concepts(
                conn,
                material_id="material-melittology",
                concepts=conn.execute(
                    "SELECT id, material_id, title, keywords_json, summary, embedding_json, created_at FROM concepts WHERE material_id = ? ORDER BY created_at ASC, id ASC",
                    ("material-melittology",),
                ).fetchall(),
                subject_tag="melittology",
                expansion={
                    "canonical_topic": "Melittology",
                    "aliases": ["Apiology"],
                    "subtopics": ["Melittology field methods", "Bee taxonomy", "Melittology or apiology bees"],
                    "related_terms": ["Apiology seminar", "Pollinator biology"],
                },
            )
            titles = {str(concept.get("title") or "").strip().lower() for concept in updated}
            self.assertIn("melittology", titles)
            self.assertIn("melittology field methods", titles)
            self.assertNotIn("apiology", titles)
            self.assertNotIn("apiology seminar", titles)
            self.assertNotIn("bee taxonomy", titles)
            self.assertNotIn("entomology", titles)
            self.assertNotIn("melittology or apiology bees", titles)
            self.assertNotIn("pollinator biology", titles)
        finally:
            conn.close()

    def test_reel_service_sync_topic_expansion_keeps_companion_keywords_for_root_topic(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO materials (
                id, subject_tag, raw_text, source_type, source_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("material-melittology-root", "melittology", "Topic: melittology", "topic", None, "2026-03-14T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO concepts (
                id, material_id, title, keywords_json, summary, embedding_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "concept-root",
                "material-melittology-root",
                "Melittology",
                '["melittology"]',
                "Core ideas, terminology, and intuition for Melittology.",
                None,
                "2026-03-14T00:00:00+00:00",
            ),
        )
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        try:
            updated = service._sync_topic_expansion_concepts(
                conn,
                material_id="material-melittology-root",
                concepts=conn.execute(
                    "SELECT id, material_id, title, keywords_json, summary, embedding_json, created_at FROM concepts WHERE material_id = ? ORDER BY created_at ASC, id ASC",
                    ("material-melittology-root",),
                ).fetchall(),
                subject_tag="melittology",
                expansion={
                    "canonical_topic": "Melittology",
                    "aliases": [],
                    "subtopics": [],
                    "related_terms": ["bees"],
                },
            )
            root_concept = next(concept for concept in updated if str(concept.get("title") or "").strip().lower() == "melittology")
            root_keywords = {str(term).strip().lower() for term in json.loads(str(root_concept.get("keywords_json") or "[]"))}
            self.assertIn("bees", root_keywords)
            titles = {str(concept.get("title") or "").strip().lower() for concept in updated}
            self.assertNotIn("bees", titles)
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

    def test_initial_response_reel_target_uses_small_bootstrap_batch(self) -> None:
        self.assertEqual(
            main_module._initial_response_reel_target(
                required_count=12,
                generation_mode="fast",
                sync_deep_fallback="if_empty",
            ),
            3,
        )
        self.assertEqual(
            main_module._initial_response_reel_target(
                required_count=12,
                generation_mode="slow",
                sync_deep_fallback="if_empty",
            ),
            5,
        )
        self.assertEqual(
            main_module._minimum_initial_response_reels(
                required_count=12,
                generation_mode="fast",
                sync_deep_fallback="if_empty",
            ),
            3,
        )
        self.assertEqual(
            main_module._minimum_initial_response_reels(
                required_count=12,
                generation_mode="slow",
                sync_deep_fallback="if_empty",
            ),
            5,
        )

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

    def test_refinement_job_excludes_source_generation_reels_from_deep_generation(self) -> None:
        conn = self._build_generation_test_conn()
        captured_excluded_generation_ids: list[str] = []

        source_generation_id = "bootstrap-gen"
        conn.execute(
            """
            INSERT INTO reel_generations (
                id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_generation_id,
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
            INSERT INTO reel_generation_heads (
                id, material_id, request_key, active_generation_id, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("head-1", "material-1", "request-key", source_generation_id, "2026-03-13T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO reel_generation_jobs (
                id, material_id, concept_id, request_key, source_generation_id, result_generation_id,
                target_profile, request_params_json, status, created_at, started_at, completed_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "material-1",
                None,
                "request-key",
                source_generation_id,
                None,
                "deep",
                "{}",
                "queued",
                "2026-03-13T00:01:00+00:00",
                None,
                None,
                None,
            ),
        )

        def fake_generate_reels(*args, **kwargs):
            captured_excluded_generation_ids[:] = list(kwargs.get("exclude_generation_ids") or [])
            generation_id = str(kwargs["generation_id"])
            conn.execute(
                """
                INSERT INTO reels (
                    id, generation_id, material_id, concept_id, video_id, video_url,
                    t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
                ) VALUES (?, ?, ?, ?, ?, '', 0, 55, '', '[]', 0, ?)
                """,
                (
                    f"{generation_id}-deep-0",
                    generation_id,
                    "material-1",
                    "concept-1",
                    "video-deep-0",
                    "2026-03-13T00:02:00+00:00",
                ),
            )
            return []

        with mock.patch.object(main_module.reel_service, "generate_reels", side_effect=fake_generate_reels):
            with mock.patch.object(main_module, "get_conn") as patched_get_conn:
                class _Ctx:
                    def __enter__(self_nonlocal):
                        return conn

                    def __exit__(self_nonlocal, exc_type, exc, tb):
                        return False

                patched_get_conn.return_value = _Ctx()
                main_module._run_refinement_job("job-1")

        self.assertEqual(captured_excluded_generation_ids, [source_generation_id])
        updated_job = conn.execute(
            "SELECT status, result_generation_id FROM reel_generation_jobs WHERE id = ?",
            ("job-1",),
        ).fetchone()
        self.assertEqual(updated_job["status"], "completed")
        self.assertTrue(str(updated_job["result_generation_id"]))
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
                sync_deep_fallback="always",
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

    def test_ensure_generation_runs_sync_deep_for_fast_mode_when_bootstrap_only_finds_one(self) -> None:
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
                sync_deep_fallback="if_empty",
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

    def test_ensure_generation_upgrades_existing_refinement_target_for_active_bootstrap(self) -> None:
        conn = self._build_generation_test_conn()
        request_key = _build_generation_request_key(
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
        conn.execute(
            """
            INSERT INTO reel_generations (
                id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bootstrap-gen",
                "material-1",
                None,
                request_key,
                "fast",
                "bootstrap",
                "active",
                None,
                3,
                "2026-03-13T00:00:00+00:00",
                "2026-03-13T00:00:00+00:00",
                "2026-03-13T00:00:00+00:00",
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO reel_generation_heads (
                id, material_id, request_key, active_generation_id, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("head-1", "material-1", request_key, "bootstrap-gen", "2026-03-13T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO reel_generation_jobs (
                id, material_id, concept_id, request_key, source_generation_id, result_generation_id,
                target_profile, request_params_json, status, created_at, started_at, completed_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "material-1",
                None,
                request_key,
                "bootstrap-gen",
                None,
                "deep",
                json.dumps({"target_reel_count": 4}),
                "queued",
                "2026-03-13T00:01:00+00:00",
                None,
                None,
                None,
            ),
        )
        conn.executemany(
            """
            INSERT INTO reels (
                id, generation_id, material_id, concept_id, video_id, video_url,
                t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
            ) VALUES (?, ?, ?, ?, ?, '', 0, 55, '', '[]', 0, ?)
            """,
            [
                ("bootstrap-reel-1", "bootstrap-gen", "material-1", "concept-1", "video-1", "2026-03-13T00:00:01+00:00"),
                ("bootstrap-reel-2", "bootstrap-gen", "material-1", "concept-1", "video-2", "2026-03-13T00:00:02+00:00"),
                ("bootstrap-reel-3", "bootstrap-gen", "material-1", "concept-1", "video-3", "2026-03-13T00:00:03+00:00"),
            ],
        )

        with mock.patch.object(main_module.reel_service, "generate_reels") as generate_reels_mock, mock.patch.object(
            main_module,
            "_ranked_request_reels",
            side_effect=lambda test_conn, **kwargs: self._fake_ranked_request_reels(test_conn, kwargs["generation_id"]),
        ):
            result = main_module._ensure_generation_for_request(
                conn,
                material_id="material-1",
                concept_id=None,
                required_count=9,
                sync_deep_fallback="if_empty",
                creative_commons_only=False,
                generation_mode="fast",
                min_relevance=None,
                video_pool_mode="short-first",
                preferred_video_duration="any",
                target_clip_duration_sec=55,
                target_clip_duration_min_sec=20,
                target_clip_duration_max_sec=55,
            )

        generate_reels_mock.assert_not_called()
        updated_job = conn.execute(
            "SELECT request_params_json FROM reel_generation_jobs WHERE id = ?",
            ("job-1",),
        ).fetchone()
        self.assertIsNotNone(updated_job)
        request_params = json.loads(str(updated_job["request_params_json"]))
        self.assertEqual(
            int(request_params["target_reel_count"]),
            main_module._refinement_target_reel_count(
                required_count=9,
                generation_mode="fast",
                existing_source_count=3,
            ),
        )
        self.assertEqual(result["generation_id"], "bootstrap-gen")
        self.assertEqual(result["response_profile"], "bootstrap")
        self.assertEqual(result["refinement_job_id"], "job-1")
        self.assertEqual(len(result["reels"]), 3)
        conn.close()

    def test_ensure_generation_extends_active_deep_generation_cumulatively(self) -> None:
        conn = self._build_generation_test_conn()
        request_key = _build_generation_request_key(
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
        conn.execute(
            """
            INSERT INTO reel_generations (
                id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bootstrap-gen",
                "material-1",
                None,
                request_key,
                "fast",
                "bootstrap",
                "active",
                None,
                4,
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
                "deep-gen",
                "material-1",
                None,
                request_key,
                "fast",
                "deep",
                "active",
                "bootstrap-gen",
                10,
                "2026-03-13T00:05:00+00:00",
                "2026-03-13T00:05:00+00:00",
                "2026-03-13T00:05:00+00:00",
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO reel_generation_heads (
                id, material_id, request_key, active_generation_id, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("head-1", "material-1", request_key, "deep-gen", "2026-03-13T00:05:00+00:00"),
        )

        active_reels = [
            {
                "reel_id": f"active-{index}",
                "material_id": "material-1",
                "concept_id": "concept-1",
                "video_url": f"https://www.youtube.com/embed/video-active-{index}?start=0&end=20",
                "t_start": 0.0,
                "t_end": 20.0,
                "video_duration_sec": 120,
                "clip_duration_sec": 20.0,
            }
            for index in range(10)
        ]
        expanded_reels = [
            {
                "reel_id": f"expanded-{index}",
                "material_id": "material-1",
                "concept_id": "concept-1",
                "video_url": f"https://www.youtube.com/embed/video-expanded-{index}?start=0&end=20",
                "t_start": 0.0,
                "t_end": 20.0,
                "video_duration_sec": 120,
                "clip_duration_sec": 20.0,
            }
            for index in range(15)
        ]
        captured: dict[str, Any] = {}
        emitted: list[dict[str, Any]] = []

        def fake_ranked_request_reels(_conn, **kwargs):
            generation_id = str(kwargs["generation_id"] or "")
            if generation_id == "deep-gen":
                return list(active_reels)
            return list(expanded_reels)

        def fake_generate_reels(*args, **kwargs):
            generation_id = str(kwargs["generation_id"])
            captured["retrieval_profile"] = str(kwargs["retrieval_profile"])
            captured["num_reels"] = int(kwargs["num_reels"])
            captured["exclude_generation_ids"] = list(kwargs.get("exclude_generation_ids") or [])
            conn.execute(
                """
                INSERT INTO reels (
                    id, generation_id, material_id, concept_id, video_id, video_url,
                    t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
                ) VALUES (?, ?, ?, ?, ?, '', 0, 55, '', '[]', 0, ?)
                """,
                (
                    f"{generation_id}-new-0",
                    generation_id,
                    "material-1",
                    "concept-1",
                    "video-new-0",
                    "2026-03-13T00:06:00+00:00",
                ),
            )
            callback = kwargs.get("on_reel_created")
            if callback is not None:
                callback(
                    {
                        "reel_id": "new-reel-0",
                        "material_id": "material-1",
                        "concept_id": "concept-1",
                        "video_url": "https://www.youtube.com/embed/video-new-0?start=0&end=55",
                        "t_start": 0.0,
                        "t_end": 55.0,
                        "video_duration_sec": 180,
                        "clip_duration_sec": 55.0,
                    }
                )
            return []

        with mock.patch.object(main_module, "_ranked_request_reels", side_effect=fake_ranked_request_reels), mock.patch.object(
            main_module.reel_service,
            "generate_reels",
            side_effect=fake_generate_reels,
        ), mock.patch.object(main_module, "_queue_refinement_job") as queue_job:
            result = main_module._ensure_generation_for_request(
                conn,
                material_id="material-1",
                concept_id=None,
                required_count=15,
                sync_deep_fallback="if_empty",
                creative_commons_only=False,
                generation_mode="fast",
                min_relevance=None,
                video_pool_mode="short-first",
                preferred_video_duration="any",
                target_clip_duration_sec=55,
                target_clip_duration_min_sec=20,
                target_clip_duration_max_sec=55,
                on_reel_created=emitted.append,
                emit_existing_reels=True,
            )

        queue_job.assert_not_called()
        self.assertEqual(captured["retrieval_profile"], "deep")
        self.assertEqual(captured["num_reels"], 15)
        self.assertEqual(captured["exclude_generation_ids"], ["bootstrap-gen", "deep-gen"])
        self.assertEqual(result["response_profile"], "deep")
        self.assertEqual(len(result["reels"]), 15)
        self.assertEqual(len(emitted), 11)
        self.assertEqual(emitted[0]["reel_id"], "active-0")
        self.assertEqual(emitted[-1]["reel_id"], "new-reel-0")
        self.assertNotEqual(result["generation_id"], "deep-gen")
        head = conn.execute(
            "SELECT active_generation_id FROM reel_generation_heads WHERE id = ?",
            (main_module._build_generation_head_id("material-1", request_key),),
        ).fetchone()
        self.assertIsNotNone(head)
        self.assertEqual(str(head["active_generation_id"]), str(result["generation_id"]))
        next_generation = conn.execute(
            "SELECT retrieval_profile, source_generation_id FROM reel_generations WHERE id = ?",
            (result["generation_id"],),
        ).fetchone()
        self.assertEqual((next_generation["retrieval_profile"], next_generation["source_generation_id"]), ("deep", "deep-gen"))
        conn.close()

    def test_feed_queues_bootstrap_refinement_with_sync_target_budget(self) -> None:
        conn = self._build_generation_test_conn()
        target_clip_duration_sec, target_clip_duration_min_sec, target_clip_duration_max_sec = _resolve_target_clip_duration_bounds(
            55,
            None,
            None,
        )
        request_key = _build_generation_request_key(
            material_id="material-1",
            concept_id=None,
            creative_commons_only=False,
            generation_mode="fast",
            video_pool_mode="short-first",
            preferred_video_duration="any",
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        conn.execute(
            """
            INSERT INTO reel_generations (
                id, material_id, concept_id, request_key, generation_mode, retrieval_profile,
                status, source_generation_id, reel_count, created_at, completed_at, activated_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bootstrap-gen",
                "material-1",
                None,
                request_key,
                "fast",
                "bootstrap",
                "active",
                None,
                5,
                "2026-03-13T00:00:00+00:00",
                "2026-03-13T00:00:00+00:00",
                "2026-03-13T00:00:00+00:00",
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO reel_generation_heads (
                id, material_id, request_key, active_generation_id, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("head-1", "material-1", request_key, "bootstrap-gen", "2026-03-13T00:00:00+00:00"),
        )

        ranked_reels = [
            {"reel_id": f"reel-{index}", "video_duration_sec": 600, "clip_duration_sec": 55.0}
            for index in range(5)
        ]

        with mock.patch.object(main_module, "_ranked_request_reels", return_value=ranked_reels), mock.patch.object(
            main_module,
            "_queue_refinement_job",
            return_value={"id": "job-feed", "status": "queued"},
        ) as queue_job, mock.patch.object(main_module, "_ensure_generation_for_request") as ensure_generation, mock.patch.object(
            main_module,
            "get_conn",
        ) as patched_get_conn:
            class _Ctx:
                def __enter__(self_nonlocal):
                    return conn

                def __exit__(self_nonlocal, exc_type, exc, tb):
                    return False

            patched_get_conn.return_value = _Ctx()
            result = main_module.feed(
                material_id="material-1",
                page=1,
                limit=1,
                autofill=True,
                prefetch=0,
                generation_mode="fast",
            )

        ensure_generation.assert_not_called()
        queue_job.assert_called_once()
        request_params = queue_job.call_args.kwargs["request_params"]
        self.assertEqual(
            int(request_params["target_reel_count"]),
            main_module._refinement_target_reel_count(
                required_count=5,
                generation_mode="fast",
                existing_source_count=5,
            ),
        )
        self.assertEqual(result["response_profile"], "bootstrap")
        self.assertEqual(result["refinement_job_id"], "job-feed")
        self.assertEqual(result["refinement_status"], "queued")
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
        self.assertTrue(all(candidate.source_surface == "youtube_html" for candidate in candidates))
        self.assertEqual(candidates[0].stage, "high_precision")
        self.assertEqual(candidates[1].stage, "high_precision")
        self.assertEqual(candidates[2].stage, "broad")
        self.assertEqual(candidates[0].text, "Binary Search Trees")
        self.assertEqual(candidates[1].text, "Binary Search Trees tutorial")
        self.assertEqual(candidates[2].text, "Binary Search Trees explained")

    def test_reel_service_bootstrap_stage_plan_skips_broad(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        candidates = [
            QueryCandidate("query one", "literal", 0.9, stage="high_precision", source_surface="youtube_html"),
            QueryCandidate("query two", "tutorial", 0.85, stage="high_precision", source_surface="youtube_html"),
            QueryCandidate("query three", "recovery_adjacent", 0.7, stage="recovery", source_surface="youtube_html"),
        ]

        plans = service._build_retrieval_stage_plan(
            query_candidates=candidates,
            fast_mode=False,
            retrieval_profile="bootstrap",
            request_need=1,
        )

        self.assertEqual([plan.name for plan in plans], ["high_precision"])
        self.assertEqual(plans[0].budget, 3)

    def test_reel_service_deep_stage_plan_expands_for_larger_request(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        candidates = [
            QueryCandidate("query one", "literal", 0.95, stage="high_precision", source_surface="youtube_html"),
            QueryCandidate("query two", "tutorial", 0.9, stage="high_precision", source_surface="youtube_html"),
            QueryCandidate("query three", "literal", 0.84, stage="broad", source_surface="youtube_html"),
            QueryCandidate("query four", "worked_example", 0.82, stage="broad", source_surface="youtube_html"),
            QueryCandidate("query five", "explained", 0.8, stage="broad", source_surface="youtube_html"),
            QueryCandidate("query six", "recovery_adjacent", 0.7, stage="recovery", source_surface="youtube_html"),
        ]

        plans = service._build_retrieval_stage_plan(
            query_candidates=candidates,
            fast_mode=False,
            retrieval_profile="deep",
            request_need=8,
        )

        self.assertEqual([plan.name for plan in plans], ["high_precision", "broad", "recovery"])
        self.assertEqual(plans[0].budget, 2)
        self.assertEqual(plans[1].budget, 3)
        self.assertEqual(plans[2].budget, 1)

    def test_reel_service_bootstrap_primary_query_disambiguates_single_term_topics(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)

        query = service._build_bootstrap_primary_query(
            title="Calculus",
            keywords=["derivatives", "integrals", "limits"],
            context_terms=["lesson"],
            subject_tag="mathematics",
        )

        self.assertEqual(query, "mathematics Calculus")

    def test_reel_service_query_planner_limits_bootstrap_concepts_by_request_need(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        concepts = [
            {"id": f"concept-{idx}", "title": f"Topic {idx}", "keywords_json": "[]", "summary": "", "created_at": f"{idx}"}
            for idx in range(5)
        ]

        plan = service._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag="computer science",
            material_context_terms=["computer science"],
            retrieval_profile="bootstrap",
            fast_mode=False,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=1,
            targeted_concept_id=None,
        )

        self.assertEqual(plan.total_selected_concepts, 2)
        self.assertTrue(plan.query_budget_exhausted)
        self.assertEqual([decision.concept_rank for decision in plan.skipped_concepts], [3, 4, 5])

    def test_reel_service_query_planner_selects_deterministic_intent_strategy(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        concepts = [
            {
                "id": "concept-math",
                "title": "Integrals",
                "keywords_json": '["practice problems", "antiderivatives"]',
                "summary": "Learn how to solve integral problems step by step.",
                "created_at": "1",
            }
        ]

        plan = service._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag="mathematics",
            material_context_terms=["mathematics"],
            retrieval_profile="bootstrap",
            fast_mode=False,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=1,
            targeted_concept_id="concept-math",
        )

        self.assertEqual(plan.selected_concepts[0].selected_intent_strategy, "worked_example")
        self.assertEqual(plan.selected_concepts[0].intent_query.text, "mathematics Integrals worked example")

    def test_reel_service_query_planner_prefers_language_disambiguation(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        concepts = [
            {
                "id": "concept-loops",
                "title": "Loops",
                "keywords_json": '["Python", "iteration"]',
                "summary": "Control flow and iteration in code.",
                "created_at": "1",
            }
        ]

        plan = service._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag="computer science",
            material_context_terms=["computer science", "python"],
            retrieval_profile="bootstrap",
            fast_mode=False,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=1,
            targeted_concept_id="concept-loops",
        )

        self.assertEqual(plan.selected_concepts[0].literal_query.text, "Python Loops")
        self.assertEqual(plan.selected_concepts[0].disambiguator, "Python")

    def test_reel_service_deep_query_planner_adds_keyword_and_alias_expansions(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        concepts = [
            {
                "id": "concept-chain-rule",
                "title": "Chain Rule",
                "keywords_json": '["composite functions", "derivatives"]',
                "summary": "Differentiate composite functions using outer and inner derivatives.",
                "created_at": "1",
            }
        ]

        plan = service._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag="mathematics",
            material_context_terms=["mathematics", "derivatives", "composite functions"],
            retrieval_profile="deep",
            fast_mode=False,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=8,
            targeted_concept_id="concept-chain-rule",
        )

        expansion_texts = [query.text for query in plan.selected_concepts[0].expansion_queries]

        self.assertIn("mathematics Chain Rule composite functions", expansion_texts)
        self.assertTrue(any("composite function derivative" in text for text in expansion_texts))
        self.assertGreaterEqual(len(plan.selected_concepts[0].recovery_queries), 1)

    def test_reel_service_deep_query_planner_breaks_broad_calculus_into_subtopics(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        concepts = [
            {
                "id": "concept-calculus",
                "title": "Calculus",
                "keywords_json": "[]",
                "summary": "Core ideas in introductory calculus.",
                "created_at": "1",
            }
        ]

        plan = service._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag="mathematics",
            material_context_terms=["mathematics", "functions"],
            retrieval_profile="deep",
            fast_mode=False,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=10,
            targeted_concept_id="concept-calculus",
        )

        expansion_texts = [query.text.lower() for query in plan.selected_concepts[0].expansion_queries]
        recovery_texts = [query.text.lower() for query in plan.selected_concepts[0].recovery_queries]

        self.assertTrue(any("limits" in text for text in expansion_texts))
        self.assertTrue(any("derivatives" in text or "chain rule" in text for text in expansion_texts))
        self.assertTrue(any("continuity" in text or "product rule" in text for text in recovery_texts))

    def test_reel_service_deep_query_planner_breaks_broad_psychology_into_subtopics(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        concepts = [
            {
                "id": "concept-psychology",
                "title": "Psychology",
                "keywords_json": "[]",
                "summary": "Core ideas in introductory psychology.",
                "created_at": "1",
            }
        ]

        plan = service._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag="psychology",
            material_context_terms=["psychology", "human behavior"],
            retrieval_profile="deep",
            fast_mode=False,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=10,
            targeted_concept_id="concept-psychology",
        )

        expansion_texts = [query.text.lower() for query in plan.selected_concepts[0].expansion_queries]
        self.assertTrue(any("cognitive psychology" in text for text in expansion_texts))
        self.assertTrue(any("behavioral psychology" in text or "social psychology" in text for text in expansion_texts))

    def test_reel_service_bootstrap_planner_can_defer_subtopic_expansion_for_broad_topic(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        concepts = [
            {
                "id": "concept-psychology",
                "title": "Psychology",
                "keywords_json": json.dumps(
                    [
                        "cognitive psychology",
                        "behavioral psychology",
                        "social psychology",
                    ]
                ),
                "summary": "Core ideas in introductory psychology.",
                "created_at": "1",
            }
        ]
        plan = service._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag="psychology",
            material_context_terms=["psychology", "human behavior"],
            retrieval_profile="bootstrap",
            fast_mode=True,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=3,
            targeted_concept_id="concept-psychology",
            allow_bootstrap_subtopic_expansion=False,
        )
        expansion_texts = [query.text.lower() for query in plan.selected_concepts[0].expansion_queries]
        self.assertFalse(any("cognitive psychology" in text or "behavioral psychology" in text for text in expansion_texts))

    def test_reel_service_deep_topic_expansion_merges_ai_terms_when_available(self) -> None:
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
        service.openai_client = mock.Mock()
        service.openai_client.chat.completions.create.return_value = mock.Mock(
            choices=[
                mock.Mock(
                    message=mock.Mock(
                        content=json.dumps(
                            {
                                "aliases": ["behavior science"],
                                "subtopics": ["attachment theory", "memory"],
                                "related_terms": ["cognitive bias"],
                            }
                        )
                    )
                )
            ]
        )
        with mock.patch.object(
            service.topic_expansion_service,
            "expand_topic",
            return_value={
                "canonical_topic": "Psychology",
                "aliases": ["Psychology"],
                "subtopics": ["cognitive psychology", "behavioral psychology"],
                "related_terms": ["classic studies in psychology"],
            },
        ):
            expansion = service._deep_topic_expansion(
                conn,
                material_id="material-ai-expand",
                subject_tag="psychology",
                generation_id=None,
            )
        self.assertIn("attachment theory", expansion["subtopics"])
        self.assertIn("memory", expansion["subtopics"])
        self.assertIn("behavior science", expansion["aliases"])
        self.assertIn("cognitive bias", expansion["related_terms"])
        conn.close()

    def test_reel_service_query_dedupe_keeps_strongest_variant(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        queries = [
            service._build_planned_query(
                text="Python loops tutorial",
                strategy="tutorial",
                stage="high_precision",
                confidence=0.9,
                source_terms=["Python loops"],
                concept_title="Python loops",
                rationale="stronger",
            ),
            service._build_planned_query(
                text=" python   loops tutorial ",
                strategy="tutorial",
                stage="high_precision",
                confidence=0.75,
                source_terms=["loops"],
                concept_title="Python loops",
                rationale="weaker duplicate",
            ),
        ]

        deduped = service._dedupe_queries(queries)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].rationale, "stronger")

    def test_reel_service_recovery_queries_are_planned_separately(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        recovery_queries = service._plan_recovery_queries(
            concept_title="Binary Search Trees",
            keywords=["tree traversal", "algorithms"],
            summary="Study insertion and search.",
            subject_tag="computer science",
            context_terms=["computer science"],
            retrieval_profile="bootstrap",
        )

        self.assertEqual(len(recovery_queries), 1)
        self.assertEqual(recovery_queries[0].stage, "recovery")
        self.assertEqual(recovery_queries[0].strategy, "recovery_adjacent")

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
            ("short", None),
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
                INSERT INTO reel_feedback (
                    id, reel_id, helpful, confusing, rating, saved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("feedback-1", "reel-ranked", 1, 0, 5, 1, "2026-03-13T00:05:00+00:00"),
            )
            third = service.ranked_feed(conn, "material-ranked", fast_mode=True)
            self.assertEqual(len(relevance_calls), 2)
            self.assertGreater(third[0]["score"], first[0]["score"])

            conn.execute(
                """
                UPDATE transcript_cache
                SET transcript_json = ?, created_at = ?
                WHERE video_id = ?
                """,
                (
                    '[{"start": 0.0, "duration": 5.0, "text": "Updated transcript"}]',
                    "2026-03-13T00:06:00+00:00",
                    "video-ranked",
                ),
            )
            service.ranked_feed(conn, "material-ranked", fast_mode=True)
            self.assertEqual(len(relevance_calls), 3)
            self.assertEqual(len(caption_calls), 3)
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
        self.assertEqual(result["specific_concept_anchor"]["hits"], [])

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

    def test_transcript_prefetch_relaunches_for_new_top_candidates(self) -> None:
        youtube_service = mock.Mock()
        service = ReelService(embedding_service=mock.Mock(), youtube_service=youtube_service)
        fetched_ids: list[str] = []

        def fake_get_transcript(_conn, video_id: str):
            fetched_ids.append(str(video_id))
            return [{"start": 0.0, "duration": 5.0, "text": f"Transcript for {video_id}"}]

        youtube_service.get_transcript.side_effect = fake_get_transcript
        candidates = [
            {"video_id": "video-a", "video_duration": 420, "ranking": {"final_score": 0.9}},
            {"video_id": "video-b", "video_duration": 510, "ranking": {"final_score": 0.8}},
            {"video_id": "video-c", "video_duration": 600, "ranking": {"final_score": 0.7}},
        ]

        task = service._maybe_launch_transcript_prefetch(
            prefetch_task=None,
            stage_candidates=candidates[:2],
            transcript_budget=2,
            clip_min_len=20,
            clip_max_len=55,
            fast_mode=True,
        )
        task = service._maybe_launch_transcript_prefetch(
            prefetch_task=task,
            stage_candidates=[candidates[0], candidates[2]],
            transcript_budget=2,
            clip_min_len=20,
            clip_max_len=55,
            fast_mode=True,
        )
        cache: dict[str, list[dict[str, object]]] = {}
        cache["video-a"] = service._transcript_for_candidate(
            prefetch_task=task,
            transcript_cache=cache,
            video_id="video-a",
            fast_mode=True,
        )
        cache["video-c"] = service._transcript_for_candidate(
            prefetch_task=task,
            transcript_cache=cache,
            video_id="video-c",
            fast_mode=True,
        )
        service._shutdown_transcript_prefetch_task(task, wait=True, cancel_futures=False)

        self.assertEqual(set(task.cached_transcripts), {"video-a", "video-c"})
        self.assertEqual(set(fetched_ids), {"video-a", "video-b", "video-c"})
        self.assertIn("video-a", cache)
        self.assertIn("video-c", cache)

    def test_transcript_for_candidate_waits_only_for_requested_future(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())

        class FakeFuture:
            def __init__(self, payload: dict[str, object]) -> None:
                self.timeout: float | None = None
                self.payload = payload

            def result(self, timeout: float | None = None):
                self.timeout = timeout
                return [self.payload]

            def done(self) -> bool:
                return True

        class FakeExecutor:
            def __init__(self) -> None:
                self.calls: list[tuple[bool, bool]] = []

            def shutdown(self, wait: bool = False, cancel_futures: bool = False) -> None:
                self.calls.append((wait, cancel_futures))

        future_a = FakeFuture({"start": 0.0, "duration": 5.0, "text": "Transcript for video-a"})
        future_b = FakeFuture({"start": 1.0, "duration": 5.0, "text": "Transcript for video-b"})
        executor = FakeExecutor()
        task = TranscriptPrefetchTask(
            video_ids=("video-a", "video-b"),
            executor=executor,
            future_by_video_id={"video-a": future_a, "video-b": future_b},
        )

        cache: dict[str, list[dict[str, object]]] = {}
        transcript = service._transcript_for_candidate(
            prefetch_task=task,
            transcript_cache=cache,
            video_id="video-a",
            fast_mode=True,
        )

        self.assertEqual(future_a.timeout, 30.0)
        self.assertIsNone(future_b.timeout)
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0]["text"], "Transcript for video-a")
        self.assertIn("video-b", task.future_by_video_id)
        self.assertEqual(executor.calls, [])

    def test_shutdown_transcript_prefetch_waits_for_executor_shutdown(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())

        class FakeFuture:
            def done(self) -> bool:
                return True

            def cancel(self) -> None:
                return None

        class FakeExecutor:
            def __init__(self) -> None:
                self.calls: list[tuple[bool, bool]] = []

            def shutdown(self, wait: bool = False, cancel_futures: bool = False) -> None:
                self.calls.append((wait, cancel_futures))

        executor = FakeExecutor()
        task = TranscriptPrefetchTask(
            video_ids=("video-a",),
            executor=executor,
            future_by_video_id={"video-a": FakeFuture()},
        )

        service._shutdown_transcript_prefetch_task(task, wait=True, cancel_futures=False)

        self.assertEqual(executor.calls, [(True, False)])
        self.assertEqual(task.future_by_video_id, {})

    def test_filter_reels_by_min_relevance_falls_back_for_niche_batches(self) -> None:
        reels = [
            {"reel_id": "a", "relevance_score": 0.12},
            {"reel_id": "b", "relevance_score": 0.11},
            {"reel_id": "c", "relevance_score": 0.09},
            {"reel_id": "d", "relevance_score": 0.04},
        ]

        filtered = main_module._filter_reels_by_min_relevance(reels, 0.3)

        self.assertEqual([row["reel_id"] for row in filtered], ["a", "b", "c"])

    def test_bootstrap_topic_keywords_include_canonical_aliases(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        concept = {
            "title": "Apiology",
            "summary": "Core ideas, terminology, and intuition for Apiology.",
        }

        with mock.patch.object(
            service.topic_expansion_service,
            "expand_topic",
            return_value={
                "canonical_topic": "Melittology",
                "aliases": ["Melittology"],
                "subtopics": [],
                "related_terms": ["Apiology"],
            },
        ):
            keywords = service._bootstrap_topic_keywords(concept, subject_tag="apiology", conn=None)

        self.assertEqual(keywords[0], "apiology")
        self.assertIn("melittology", keywords)

    def test_bootstrap_topic_keywords_prioritize_companion_terms_for_opaque_topics(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        concept = {
            "title": "Apiology",
            "summary": "Core ideas, terminology, and intuition for Apiology.",
        }

        with mock.patch.object(
            service.topic_expansion_service,
            "expand_topic",
            return_value={
                "canonical_topic": "Melittology",
                "aliases": ["Melittology"],
                "subtopics": [],
                "related_terms": ["bees", "Melittology"],
            },
        ):
            keywords = service._bootstrap_topic_keywords(concept, subject_tag="apiology", conn=None)

        self.assertEqual(keywords[:3], ["apiology", "bees", "melittology"])

    def test_choose_disambiguator_does_not_repeat_subject_for_opaque_root_topic(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())
        disambiguator = service._choose_disambiguator(
            title="Melittology",
            keywords=["melittology", "bees", "melittology explained"],
            context_terms=[],
            subject_tag="melittology",
        )
        literal_query = service._build_literal_query(
            title="Melittology",
            keywords=["melittology", "bees", "melittology explained"],
            disambiguator=disambiguator,
        )
        self.assertEqual(disambiguator, "bees")
        self.assertEqual(literal_query, "bees Melittology")

    def test_bootstrap_query_expansion_uses_standalone_alias_for_niche_topic(self) -> None:
        service = ReelService(embedding_service=mock.Mock(), youtube_service=mock.Mock())

        expansions = service._maybe_expand_queries(
            concept_title="Apiology",
            keywords=["apiology", "melittology", "entomology"],
            summary="The scientific study of bees and related bee behavior within entomology.",
            context_terms=[],
            literal_query="apiology melittology",
            intent_plan=ConceptIntentPlan(strategy="explained", suffix="explained", rationale=""),
            retrieval_profile="bootstrap",
            fast_mode=True,
            subject_tag="apiology",
            disambiguator=None,
            request_need=5,
            allow_bootstrap_subtopic_expansion=True,
        )

        self.assertTrue(any(query.text == "melittology explained" for query in expansions))

    def test_ensure_generation_for_request_forwards_incremental_reel_callback(self) -> None:
        conn = self._build_generation_test_conn()
        emitted: list[dict[str, object]] = []
        emitted_reel = {
            "reel_id": "generated-reel-1",
            "concept_id": "concept-1",
            "video_url": "https://www.youtube.com/embed/video-1?start=0&end=30",
            "t_start": 0.0,
            "t_end": 30.0,
            "video_duration_sec": 180,
            "clip_duration_sec": 30.0,
            "relevance_score": 0.82,
            "discovery_score": 0.68,
        }

        def fake_generate_reels(*args, **kwargs):
            generation_id = str(kwargs["generation_id"])
            callback = kwargs.get("on_reel_created")
            conn.execute(
                """
                INSERT INTO reels (
                    id, generation_id, material_id, concept_id, video_id, video_url,
                    t_start, t_end, transcript_snippet, takeaways_json, base_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    emitted_reel["reel_id"],
                    generation_id,
                    "material-1",
                    "concept-1",
                    "video-1",
                    emitted_reel["video_url"],
                    emitted_reel["t_start"],
                    emitted_reel["t_end"],
                    "",
                    "[]",
                    0.8,
                    "2026-03-13T00:00:00+00:00",
                ),
            )
            if callback is not None:
                callback(dict(emitted_reel))
            return []

        with mock.patch.object(main_module.reel_service, "generate_reels", side_effect=fake_generate_reels), mock.patch.object(
            main_module,
            "_ranked_request_reels",
            return_value=[dict(emitted_reel)],
        ), mock.patch.object(main_module, "_queue_refinement_job", return_value=None):
            result = main_module._ensure_generation_for_request(
                conn,
                material_id="material-1",
                concept_id=None,
                required_count=1,
                creative_commons_only=False,
                generation_mode="fast",
                min_relevance=None,
                video_pool_mode="short-first",
                preferred_video_duration="any",
                target_clip_duration_sec=55,
                target_clip_duration_min_sec=20,
                target_clip_duration_max_sec=55,
                on_reel_created=emitted.append,
            )

        self.assertEqual(emitted, [emitted_reel])
        self.assertEqual(result["reels"], [emitted_reel])
        conn.close()

    def test_generate_stream_endpoint_emits_reels_then_done(self) -> None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE materials (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO materials (id) VALUES (?)", ("material-1",))

        class _ConnCtx:
            def __enter__(self_nonlocal):
                return conn

            def __exit__(self_nonlocal, exc_type, exc, tb):
                return False

        streamed_reels = [
            {
                "reel_id": "stream-reel-1",
                "concept_id": "concept-1",
                "video_url": "https://www.youtube.com/embed/video-1?start=0&end=20",
                "t_start": 0.0,
                "t_end": 20.0,
                "video_duration_sec": 120,
                "clip_duration_sec": 20.0,
            },
            {
                "reel_id": "stream-reel-2",
                "concept_id": "concept-1",
                "video_url": "https://www.youtube.com/embed/video-2?start=10&end=35",
                "t_start": 10.0,
                "t_end": 35.0,
                "video_duration_sec": 160,
                "clip_duration_sec": 25.0,
            },
        ]

        def fake_ensure_generation(*args, **kwargs):
            callback = kwargs.get("on_reel_created")
            for reel in streamed_reels:
                if callback is not None:
                    callback(dict(reel))
            return {
                "reels": list(streamed_reels),
                "generation_id": "gen-stream",
                "response_profile": "bootstrap",
                "refinement_job_id": "job-stream",
                "refinement_status": "queued",
            }

        with mock.patch.object(main_module, "get_conn", return_value=_ConnCtx()), mock.patch.object(
            main_module,
            "_ensure_generation_for_request",
            side_effect=fake_ensure_generation,
        ):
            client = TestClient(main_module.app)
            response = client.post(
                "/api/reels/generate-stream",
                json={
                    "material_id": "material-1",
                    "num_reels": 2,
                    "generation_mode": "fast",
                },
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        self.assertEqual([event["type"] for event in events], ["reel", "reel", "done"])
        self.assertEqual(events[0]["reel"]["reel_id"], "stream-reel-1")
        self.assertEqual(events[1]["reel"]["reel_id"], "stream-reel-2")
        self.assertEqual(events[2]["response"]["generation_id"], "gen-stream")
        self.assertEqual(events[2]["response"]["refinement_job_id"], "job-stream")
        conn.close()

    def test_refinement_status_uses_separate_high_volume_rate_limit_scope(self) -> None:
        conn = self._build_generation_test_conn()
        conn.execute(
            """
            INSERT INTO reel_generation_jobs (
                id, material_id, concept_id, request_key, source_generation_id, result_generation_id,
                target_profile, request_params_json, status, created_at, started_at, completed_at, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-status",
                "material-1",
                None,
                "request-key",
                "source-gen",
                None,
                "deep",
                "{}",
                "queued",
                "2026-03-14T00:00:00+00:00",
                None,
                None,
                None,
            ),
        )
        captured_limits: list[tuple[str, int]] = []

        class _ConnCtx:
            def __enter__(self_nonlocal):
                return conn

            def __exit__(self_nonlocal, exc_type, exc, tb):
                return False

        with mock.patch.object(main_module, "get_conn", return_value=_ConnCtx()), mock.patch.object(
            main_module,
            "_enforce_rate_limit",
            side_effect=lambda request, scope, *, limit, window_sec=main_module.RATE_LIMIT_WINDOW_SEC: captured_limits.append(
                (scope, limit)
            ),
        ):
            result = main_module.refinement_status(mock.Mock(), "job-status")

        self.assertEqual(
            captured_limits,
            [("reels-refinement-status", main_module.REELS_REFINEMENT_STATUS_RATE_LIMIT_PER_WINDOW)],
        )
        self.assertEqual(result["job_id"], "job-status")
        conn.close()


if __name__ == "__main__":
    unittest.main()
