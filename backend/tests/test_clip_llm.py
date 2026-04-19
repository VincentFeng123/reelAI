"""
Phase 3 tests for `backend/app/services/clip_llm.py::rerank_clip_candidates_llm`.

Covered:
  * `_parse_rerank_response` — malformed JSON, out-of-range indices, dedup
  * `_heuristic_rerank_fallback` — deterministic weighting, returns non-empty
    whenever candidates is non-empty
  * `_render_candidates_for_rerank` — no raw transcript text leaks into the
    prompt (only opener/closer/duration/sub-scores)
  * End-to-end `rerank_clip_candidates_llm` with all LLM providers
    monkeypatched to None — falls through to heuristic, returns target_count
    picks sorted by combined score desc
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services import clip_llm  # noqa: E402
from backend.app.services.clip_boundary import ClipCandidate  # noqa: E402
from backend.app.services.clip_llm import (  # noqa: E402
    RankedClipPick,
    _build_rerank_system_prompt,
    _heuristic_rerank_fallback,
    _parse_rerank_response,
    _render_candidates_for_rerank,
    rerank_clip_candidates_llm,
)


def _mk_cand(
    *,
    t_start: float,
    t_end: float,
    relevance: float = 0.5,
    engagement: float = 0.3,
    completeness: float = 0.7,
    boundary_conf: float = 0.8,
    hook: str | None = None,
    opener: str = "Let me tell you something important.",
    closer: str = "That's the core idea.",
    combined: float = 0.0,
) -> ClipCandidate:
    return ClipCandidate(
        t_start=t_start,
        t_end=t_end,
        opener_text=opener,
        closer_text=closer,
        relevance_score=relevance,
        engagement_score=engagement,
        completeness_score=completeness,
        boundary_confidence=boundary_conf,
        combined_score=combined,
        hook_pattern=hook,
        start_sentence_idx=0,
        end_sentence_idx=0,
    )


class ParseRerankResponseTests(unittest.TestCase):
    def test_valid_payload(self):
        raw = (
            '{"picks": ['
            '{"candidate_idx": 0, "virality_score": 0.9, "hook_pattern": "contradiction", "reason": "strong opener"},'
            '{"candidate_idx": 2, "virality_score": 0.6, "hook_pattern": null, "reason": "solid payoff"}'
            ']}'
        )
        picks = _parse_rerank_response(raw, num_candidates=5, target_count=3)
        self.assertEqual(len(picks), 2)
        self.assertEqual(picks[0].candidate_idx, 0)
        self.assertEqual(picks[0].virality_score, 0.9)
        self.assertEqual(picks[0].hook_pattern, "contradiction")
        self.assertIsNone(picks[1].hook_pattern)

    def test_malformed_json_returns_empty(self):
        self.assertEqual(_parse_rerank_response("not json", num_candidates=3, target_count=3), [])
        self.assertEqual(_parse_rerank_response("[]", num_candidates=3, target_count=3), [])

    def test_out_of_range_indices_dropped(self):
        raw = '{"picks": [{"candidate_idx": 99, "virality_score": 0.8, "reason": "x"}]}'
        self.assertEqual(_parse_rerank_response(raw, num_candidates=5, target_count=3), [])

    def test_duplicate_indices_deduped(self):
        raw = (
            '{"picks": ['
            '{"candidate_idx": 1, "virality_score": 0.8, "reason": "first"},'
            '{"candidate_idx": 1, "virality_score": 0.5, "reason": "duplicate"}'
            ']}'
        )
        picks = _parse_rerank_response(raw, num_candidates=5, target_count=3)
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0].virality_score, 0.8)

    def test_target_count_clamps_output(self):
        raw = '{"picks": ['
        raw += ",".join(
            f'{{"candidate_idx": {i}, "virality_score": 0.1, "reason": "n"}}'
            for i in range(10)
        )
        raw += "]}"
        picks = _parse_rerank_response(raw, num_candidates=10, target_count=3)
        self.assertEqual(len(picks), 3)

    def test_virality_clamped_to_unit_interval(self):
        raw = '{"picks": [{"candidate_idx": 0, "virality_score": 2.5, "reason": ""}]}'
        picks = _parse_rerank_response(raw, num_candidates=2, target_count=3)
        self.assertEqual(picks[0].virality_score, 1.0)
        raw = '{"picks": [{"candidate_idx": 0, "virality_score": -1.0, "reason": ""}]}'
        picks = _parse_rerank_response(raw, num_candidates=2, target_count=3)
        self.assertEqual(picks[0].virality_score, 0.0)


class HeuristicFallbackTests(unittest.TestCase):
    def test_weighting_prefers_high_relevance(self):
        # Two candidates — one high relevance, one low. With 0.45 weight on
        # relevance and 0.25 on engagement, high-relevance wins when their
        # other scores are tied.
        strong = _mk_cand(t_start=0.0, t_end=30.0, relevance=1.0, engagement=0.2)
        weak = _mk_cand(t_start=30.0, t_end=60.0, relevance=0.1, engagement=0.2)
        picks = _heuristic_rerank_fallback([weak, strong], target_count=2)
        self.assertEqual(picks[0].candidate_idx, 1)  # strong
        self.assertEqual(picks[1].candidate_idx, 0)  # weak

    def test_non_empty_for_non_empty_candidates(self):
        cand = _mk_cand(t_start=0.0, t_end=30.0)
        picks = _heuristic_rerank_fallback([cand], target_count=5)
        self.assertEqual(len(picks), 1)
        self.assertIsInstance(picks[0], RankedClipPick)

    def test_empty_input_empty_output(self):
        self.assertEqual(_heuristic_rerank_fallback([], target_count=3), [])


class PromptRenderingTests(unittest.TestCase):
    def test_render_contains_sub_scores_not_transcript(self):
        cand = _mk_cand(
            t_start=12.5, t_end=45.0, relevance=0.82, engagement=0.41,
            completeness=0.65, hook="contradiction",
            opener="But everyone believes the opposite.",
            closer="That's why I changed my mind.",
        )
        rendered = _render_candidates_for_rerank([cand])
        self.assertIn("contradiction", rendered)
        self.assertIn("rel=0.82", rendered)
        self.assertIn("eng=0.41", rendered)
        self.assertIn("cmp=0.65", rendered)
        self.assertIn("But everyone", rendered)      # opener included
        self.assertIn("changed my mind", rendered)    # closer included

    def test_system_prompt_mentions_target_count_and_hook_taxonomy(self):
        prompt = _build_rerank_system_prompt(query="cold plunge", target_count=5)
        self.assertIn("5", prompt)
        self.assertIn("cold plunge", prompt)
        self.assertIn("contradiction", prompt)
        self.assertIn("candidate_idx", prompt)


class EndToEndFallbackTests(unittest.TestCase):
    def test_no_llm_providers_falls_through_to_heuristic(self):
        cands = [
            _mk_cand(t_start=0.0, t_end=30.0, relevance=0.9, engagement=0.3),
            _mk_cand(t_start=30.0, t_end=60.0, relevance=0.3, engagement=0.2),
            _mk_cand(t_start=60.0, t_end=90.0, relevance=0.5, engagement=0.6),
        ]
        # Patch provider builders to return None — forces heuristic fallback.
        with mock.patch.object(clip_llm, "_build_gemini_client", return_value=None), \
             mock.patch.object(clip_llm, "_build_groq_client", return_value=None), \
             mock.patch.object(clip_llm, "_build_cerebras_client", return_value=None):
            picks = rerank_clip_candidates_llm(cands, query="test", target_count=2)
        self.assertEqual(len(picks), 2)
        # Heuristic weighting should rank high-relevance candidate first.
        self.assertEqual(picks[0].candidate_idx, 0)

    def test_empty_candidates_returns_empty(self):
        self.assertEqual(rerank_clip_candidates_llm([], target_count=5), [])


if __name__ == "__main__":
    unittest.main()
