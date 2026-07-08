"""
Pure unit tests for ReelService._concept_topic_query (Task T3).

No DB, no network.  ReelService accepts embedding_service=None,
youtube_service=None (pattern established by test_reels_saliency.py).
VERCEL=1 sets serverless_mode=True inside __init__, avoiding any outbound
calls from llm_router / TopicExpansionService / ProviderRegistry.

The query is the concept's CLEAN TITLE, nothing more: wiki-keyword appending
was removed — the clip engine's discover() expansion (spellcheck + field
inference) owns disambiguation, and appended keywords polluted seed queries.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.app.services.reels as reels_module  # noqa: E402
from backend.app.services.reels import ReelService  # noqa: E402


def _svc() -> ReelService:
    return ReelService(embedding_service=None, youtube_service=None)


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


class CorrectedSubjectTagTests(unittest.TestCase):
    """_corrected_subject_tag: conservative spellfix only, cached on success,
    NOT cached on transient failure."""

    def setUp(self) -> None:
        ReelService._SUBJECT_CORRECTION_CACHE.clear()
        self.addCleanup(ReelService._SUBJECT_CORRECTION_CACHE.clear)

    def _patch_expand(self, **kwargs):
        return mock.patch.object(
            reels_module._clip_engine_expand, "expand_query", **kwargs
        )

    def test_conservative_spellfix_accepted(self) -> None:
        svc = _svc()
        with self._patch_expand(
            return_value={"corrected": "psychology", "queries": ["psychology"]}
        ):
            self.assertEqual(svc._corrected_subject_tag("pychology"), "psychology")

    def test_field_qualifying_rewrite_rejected(self) -> None:
        svc = _svc()
        with self._patch_expand(
            return_value={"corrected": "jaguar animal biology", "queries": []}
        ):
            self.assertEqual(svc._corrected_subject_tag("jaguar"), "jaguar")

    def test_dissimilar_single_word_rewrite_rejected(self) -> None:
        svc = _svc()
        with self._patch_expand(return_value={"corrected": "chemistry", "queries": []}):
            self.assertEqual(svc._corrected_subject_tag("physics"), "physics")

    def test_correction_cached_after_success(self) -> None:
        svc = _svc()
        with self._patch_expand(
            return_value={"corrected": "psychology", "queries": []}
        ):
            svc._corrected_subject_tag("pychology")
        with self._patch_expand(side_effect=AssertionError("must not be called")):
            self.assertEqual(svc._corrected_subject_tag("pychology"), "psychology")

    def test_transient_failure_not_cached(self) -> None:
        svc = _svc()
        with self._patch_expand(side_effect=RuntimeError("gemini down")):
            self.assertEqual(svc._corrected_subject_tag("pychology"), "pychology")
        # the failed lookup must NOT poison the cache — the next call retries
        with self._patch_expand(
            return_value={"corrected": "psychology", "queries": []}
        ):
            self.assertEqual(svc._corrected_subject_tag("pychology"), "psychology")


if __name__ == "__main__":
    unittest.main()
