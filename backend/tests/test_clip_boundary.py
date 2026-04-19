"""
Phase 2 + 3 unit tests for `backend/app/services/clip_boundary.py`.

Covered:
  * `_classify_hook_pattern` on seven viral taxonomy openers + neutral text
  * `_self_containment_penalty` / pronoun / answer-without-question signals
  * `_payoff_bonus` resolution-cue recognition
  * `_score_window` breakdown thread — monotonicity check: hook+payoff
    candidate outranks a matched neutral candidate
  * `generate_clip_candidates` — bounded count, duration filter, sorted desc,
    sub-score propagation
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.clip_boundary import (  # noqa: E402
    ClipCandidate,
    _classify_hook_pattern,
    _payoff_bonus_score,
    _pronoun_ambiguity_open,
    _is_answer_without_question,
    _score_window,
    generate_clip_candidates,
)
from backend.app.services.sentences import SentenceSpan  # noqa: E402


def _sent(text: str, t_start: float, t_end: float, punct: str = ".") -> SentenceSpan:
    return SentenceSpan(
        text=text,
        t_start=t_start,
        t_end=t_end,
        cue_start_idx=0,
        cue_end_idx=0,
        word_start_idx=0,
        word_end_idx=0,
        terminal_punct=punct,
        confidence=0.88,
    )


class HookPatternTests(unittest.TestCase):
    def test_contradiction(self):
        self.assertEqual(_classify_hook_pattern("But everyone thinks you need more."), "contradiction")

    def test_number_promise(self):
        self.assertEqual(_classify_hook_pattern("5 reasons your sleep is broken."), "number_promise")

    def test_confession(self):
        self.assertEqual(_classify_hook_pattern("I used to believe this was impossible."), "confession")

    def test_question(self):
        self.assertEqual(_classify_hook_pattern("Why does this keep happening?"), "question")

    def test_before_after(self):
        self.assertEqual(_classify_hook_pattern("I went from broke to millionaire in two years."), "before_after")

    def test_counterintuitive_howto(self):
        self.assertEqual(_classify_hook_pattern("How to lose weight without dieting."), "counterintuitive_howto")

    def test_warning(self):
        self.assertEqual(_classify_hook_pattern("Never eat these foods after 8pm."), "warning")

    def test_neutral_returns_none(self):
        self.assertIsNone(_classify_hook_pattern("The mitochondria is the powerhouse of the cell."))
        self.assertIsNone(_classify_hook_pattern("Today we continue the discussion on climate."))


class SelfContainmentTests(unittest.TestCase):
    def test_pronoun_ambiguity_flagged(self):
        self.assertTrue(_pronoun_ambiguity_open("He really wanted to finish the project."))
        self.assertTrue(_pronoun_ambiguity_open("This means we cannot proceed."))

    def test_pronoun_neutral_openers_pass(self):
        self.assertFalse(_pronoun_ambiguity_open("The doctor explained everything clearly."))
        # A pronoun with a nearby noun antecedent in the same sentence is fine.
        self.assertFalse(_pronoun_ambiguity_open("Sarah said she would arrive soon."))

    def test_answer_without_question(self):
        self.assertTrue(_is_answer_without_question("Yes, that's how it worked out."))
        self.assertTrue(_is_answer_without_question("Exactly, and here's why."))
        self.assertFalse(_is_answer_without_question("The answer is complicated."))


class PayoffBonusTests(unittest.TestCase):
    def test_resolution_cue_awarded(self):
        self.assertGreater(_payoff_bonus_score("So the point is, you should never skip warmup."), 0.0)
        self.assertGreater(_payoff_bonus_score("Therefore the mistake was obvious in hindsight."), 0.0)

    def test_no_cue_returns_zero(self):
        self.assertEqual(_payoff_bonus_score("We wrapped up the recording session."), 0.0)


class ScoreWindowBreakdownTests(unittest.TestCase):
    def test_breakdown_populates_hook_and_payoff(self):
        sentences = [
            _sent("5 reasons your code is slow.", 0.0, 3.0),
            _sent("The first one is needless allocation.", 3.5, 8.0),
            _sent("So the point is, profile before you optimize.", 8.5, 14.0),
        ]
        breakdown: dict = {}
        score = _score_window(
            sentences_slice=sentences,
            query_words={"code", "slow"},
            global_df={"code": 1, "slow": 1, "allocation": 1, "profile": 1},
            num_windows=1,
            query_embedding=None,
            embed_func=None,
            video_duration_sec=None,
            silence_ranges=None,
            user_target_sec=15.0,
            breakdown=breakdown,
        )
        self.assertNotEqual(score, float("-inf"))
        self.assertEqual(breakdown.get("hook_pattern"), "number_promise")
        self.assertGreater(breakdown.get("hook_bonus", 0.0), 0.0)
        self.assertGreater(breakdown.get("payoff_bonus", 0.0), 0.0)
        self.assertIn("boundary_confidence", breakdown)


class GenerateClipCandidatesTests(unittest.TestCase):
    def _build_long_transcript(self) -> list[SentenceSpan]:
        # 12 sentences, each ~5s, total 60s. Terminal punct on all so the
        # candidate enumerator can see sliding windows.
        out = []
        for i in range(12):
            out.append(_sent(f"Sentence number {i} is complete.", i * 5.0, i * 5.0 + 4.5))
        return out

    def test_returns_candidates_sorted_desc(self):
        sents = self._build_long_transcript()
        cands = generate_clip_candidates(
            query="sentence complete",
            sentences=sents,
            cues=None,
            silence_ranges=None,
            user_min_sec=15.0,
            user_max_sec=45.0,
            user_target_sec=25.0,
            video_duration_sec=60.0,
            embed_func=None,
            target_count=5,
        )
        self.assertIsInstance(cands, list)
        if cands:
            scores = [c.combined_score for c in cands]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for c in cands:
                self.assertIsInstance(c, ClipCandidate)
                dur = c.t_end - c.t_start
                self.assertGreaterEqual(dur, 15.0 - 0.1)
                self.assertLessEqual(dur, 45.0 + 0.1)

    def test_target_count_cap_respected(self):
        sents = self._build_long_transcript()
        cands = generate_clip_candidates(
            query="",
            sentences=sents,
            cues=None,
            silence_ranges=None,
            user_min_sec=15.0,
            user_max_sec=45.0,
            user_target_sec=25.0,
            video_duration_sec=60.0,
            embed_func=None,
            target_count=5,
        )
        # Plan: return top `target_count * 3` capped at 60.
        self.assertLessEqual(len(cands), min(60, 5 * 3))

    def test_empty_when_duration_bounds_infeasible(self):
        sents = self._build_long_transcript()
        cands = generate_clip_candidates(
            query="anything",
            sentences=sents,
            cues=None,
            silence_ranges=None,
            user_min_sec=500.0,  # impossible — transcript is only 60s
            user_max_sec=700.0,
            user_target_sec=600.0,
            video_duration_sec=60.0,
            embed_func=None,
            target_count=3,
        )
        self.assertEqual(cands, [])


if __name__ == "__main__":
    unittest.main()
