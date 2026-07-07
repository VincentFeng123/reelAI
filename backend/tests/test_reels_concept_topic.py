"""
Pure unit tests for ReelService._concept_topic_query (Task T3).

No DB, no network.  ReelService accepts embedding_service=None,
youtube_service=None (pattern established by test_reels_saliency.py).
VERCEL=1 sets serverless_mode=True inside __init__, avoiding any outbound
calls from llm_router / TopicExpansionService / ProviderRegistry.

Note on "single-token title":
  normalize_terms returns stems + variants.  "Osmosis" → {'osmosi','osmosis'}
  (len=2) so it does NOT qualify as single-token by that rule.  Use acronyms
  like "ATP" (→ {'atp'}, len=1) to hit the keyword-append branch.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.reels import ReelService  # noqa: E402


def _svc() -> ReelService:
    return ReelService(embedding_service=None, youtube_service=None)


def _row(title: str, keywords: list[str]) -> dict:
    return {"title": title, "keywords_json": json.dumps(keywords)}


class ConceptTopicQueryTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Test 1: single-token title → first differing keyword appended
    # normalize_terms(['ATP']) == {'atp'}, len=1 → single-token branch
    # 'cell biology' normalized ('cell biology') != 'atp' → appended
    # ------------------------------------------------------------------
    def test_single_token_title_appends_first_differing_keyword(self) -> None:
        svc = _svc()
        result = svc._concept_topic_query(_row("ATP", ["cell biology", "synthesis"]))
        self.assertEqual(result, "ATP cell biology")

    # ------------------------------------------------------------------
    # Test 2: multi-token title → no keyword appended
    # normalize_terms(['Cellular respiration stages']) has len>1
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

    # ------------------------------------------------------------------
    # Test 4: keyword identical (normalized) to single-token title is skipped
    # 'atp' normalized == 'atp' (title) → skipped; 'synthesis' != 'atp' → used
    # ------------------------------------------------------------------
    def test_identical_keyword_skipped_uses_next(self) -> None:
        svc = _svc()
        result = svc._concept_topic_query(_row("ATP", ["atp", "synthesis"]))
        self.assertEqual(result, "ATP synthesis")

    # ------------------------------------------------------------------
    # Bonus: ingestion_pipeline param is stored
    # ------------------------------------------------------------------
    def test_ingestion_pipeline_stored(self) -> None:
        sentinel = object()
        svc = ReelService(embedding_service=None, youtube_service=None, ingestion_pipeline=sentinel)
        self.assertIs(svc.ingestion_pipeline, sentinel)

    def test_ingestion_pipeline_defaults_none(self) -> None:
        svc = _svc()
        self.assertIsNone(svc.ingestion_pipeline)


if __name__ == "__main__":
    unittest.main()
