"""
Pure unit tests for ReelService._concept_topic_query (Task T3).

No DB, no network.  ReelService accepts embedding_service=None,
youtube_service=None (pattern established by test_reels_saliency.py).
VERCEL=1 sets serverless_mode=True inside __init__, avoiding any outbound
calls from llm_router / TopicExpansionService / ProviderRegistry.

The query is the concept's CLEAN TITLE, nothing more: wiki-keyword appending
was removed. Topic-material concepts are produced by the shared cached AI
query plan, and appended keywords polluted seed queries.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.reels import ReelService  # noqa: E402


def _svc(*, ingestion_pipeline=None) -> ReelService:
    with mock.patch.dict(os.environ, {"VERCEL": "1"}):
        return ReelService(
            embedding_service=None,
            youtube_service=None,
            ingestion_pipeline=ingestion_pipeline,
        )


def _row(title: str, keywords: list[str]) -> dict:
    return {"title": title, "keywords_json": json.dumps(keywords)}


class ConceptTopicQueryTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Test 1: single-token title → clean title only, keywords IGNORED
    # (keyword appending removed; discover() expansion disambiguates)
    # ------------------------------------------------------------------
    def test_single_token_title_ignores_keywords(self) -> None:
        svc = _svc()
        result = svc._concept_topic_query(_row("ATP", ["cell biology", "synthesis"]))
        self.assertEqual(result, "ATP")

    def test_short_leaf_title_keeps_parent_material_context(self) -> None:
        svc = _svc()
        result = svc._concept_topic_query(
            _row("Atp", ["atp"]),
            parent_topic="cellular respiration",
        )
        self.assertEqual(result, "Atp in cellular respiration")

    def test_explicit_acronym_title_keeps_parent_material_context(self) -> None:
        svc = _svc()
        self.assertEqual(
            svc._concept_topic_query(
                _row("ATP", ["atp"]),
                parent_topic="cellular respiration",
            ),
            "ATP in cellular respiration",
        )

    def test_ordinary_short_words_do_not_gain_parent_context(self) -> None:
        svc = _svc()
        for title in ("Force", "Cell", "Atom", "Logic", "Rome"):
            with self.subTest(title=title):
                self.assertEqual(
                    svc._concept_topic_query(
                        _row(title, [title.lower()]),
                        parent_topic="introductory material",
                    ),
                    title,
                )

    def test_parent_context_is_not_duplicated(self) -> None:
        svc = _svc()
        result = svc._concept_topic_query(
            _row("ATP in cellular respiration", ["atp"]),
            parent_topic="cellular respiration",
        )
        self.assertEqual(result, "ATP in cellular respiration")

    # ------------------------------------------------------------------
    # Test 2: multi-token title → clean title only
    # ------------------------------------------------------------------
    def test_multi_token_title_no_keyword_appended(self) -> None:
        svc = _svc()
        result = svc._concept_topic_query(_row("Cellular respiration stages", ["ATP", "mitochondria"]))
        self.assertEqual(result, "Cellular respiration stages")

    # ------------------------------------------------------------------
    # Test 3: empty / blank title → ""
    # ------------------------------------------------------------------
    def test_empty_title_returns_empty_string(self) -> None:
        svc = _svc()
        self.assertEqual(svc._concept_topic_query(_row("", ["diffusion"])), "")
        self.assertEqual(svc._concept_topic_query(_row("   ", ["diffusion"])), "")

    def test_source_context_is_bounded_and_domain_agnostic(self) -> None:
        svc = _svc()
        summaries = {
            "physics": "Relate net force to acceleration and solve incline problems.",
            "biology": "Trace glycolysis through ATP production in respiration.",
            "math": "Derive the chain rule and apply it to composite functions.",
            "software": "Explain quicksort partitioning and trace recursive calls.",
            "law": "Apply duty, breach, causation, and damages to a fact pattern.",
        }
        for domain, summary in summaries.items():
            with self.subTest(domain=domain):
                self.assertEqual(
                    svc._concept_source_context(
                        {"summary": f"  {summary}\n"},
                        strict_topic_only=False,
                    ),
                    summary,
                )

        self.assertIsNone(
            svc._concept_source_context(
                {"summary": summaries["physics"]},
                strict_topic_only=True,
            )
        )
        self.assertIsNone(
            svc._concept_source_context(
                {"summary": " \n "},
                strict_topic_only=False,
            )
        )
        bounded = svc._concept_source_context(
            {"summary": "context " * 120},
            strict_topic_only=False,
        )
        self.assertIsNotNone(bounded)
        self.assertLessEqual(len(bounded), 600)
        self.assertFalse(bounded.endswith("contex"))

    # ------------------------------------------------------------------
    # Bonus: ingestion_pipeline param is stored
    # ------------------------------------------------------------------
    def test_ingestion_pipeline_stored(self) -> None:
        sentinel = object()
        svc = _svc(ingestion_pipeline=sentinel)
        self.assertIs(svc.ingestion_pipeline, sentinel)

    def test_ingestion_pipeline_defaults_none(self) -> None:
        svc = _svc()
        self.assertIsNone(svc.ingestion_pipeline)

if __name__ == "__main__":
    unittest.main()
