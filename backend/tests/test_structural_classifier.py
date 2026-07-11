"""Unit tests for ``backend/app/services/structural_classifier.py``.

Cases are grounded in real audit failures from ``audit_after_PR5.csv``:
  * 3B1B "Hey everyone, Grant here" greeting at 14.98 s of a 1027 s video
  * Patreon outro at 970 s of the same video
  * Mid-derivation passages that must NOT be flagged as structural
  * Position-aware: same phrase early vs late shifts the prior
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.structural_classifier import (  # noqa: E402
    classify_passage,
    label_penalty,
)


WUVT_DURATION_SEC = 1027.0  # 3B1B "Essence of Calculus, Chapter 1"


class IntroDetection(unittest.TestCase):
    def test_3b1b_grant_greeting_at_video_start(self) -> None:
        label = classify_passage(
            "Hey everyone, Grant here. Today's video is the start of a new series.",
            t_start=14.98,
            video_duration=WUVT_DURATION_SEC,
        )
        self.assertEqual(label.name, "intro")
        self.assertGreater(label.confidence, 0.9)

    def test_welcome_back_intro(self) -> None:
        label = classify_passage(
            "Welcome back to the channel. Today we're going to talk about derivatives.",
            t_start=5.0,
            video_duration=600.0,
        )
        self.assertEqual(label.name, "intro")

    def test_intro_phrase_late_in_video_is_quoted_not_intro(self) -> None:
        label = classify_passage(
            "A lot of channels start with hey everyone today we're going to talk about gradient descent.",
            t_start=500.0,
            video_duration=600.0,
        )
        self.assertEqual(label.name, "substantive")

    def test_topic_preamble_alone(self) -> None:
        label = classify_passage(
            "In today's video we're going to learn about Fourier transforms.",
            t_start=8.0,
            video_duration=900.0,
        )
        self.assertEqual(label.name, "intro")


class OutroDetection(unittest.TestCase):
    def test_thanks_for_watching_at_end(self) -> None:
        label = classify_passage(
            "Thanks for watching, and I'll see you in the next one.",
            t_start=580.0,
            video_duration=600.0,
        )
        self.assertEqual(label.name, "outro")
        self.assertGreater(label.confidence, 0.9)

    def test_subscribe_cta_at_end(self) -> None:
        label = classify_passage(
            "If you enjoyed this video, please don't forget to like and subscribe.",
            t_start=950.0,
            video_duration=1000.0,
        )
        self.assertEqual(label.name, "outro")

    def test_outro_phrase_early_in_video_is_not_outro(self) -> None:
        label = classify_passage(
            "Thanks for watching this intro is what most channels lead with.",
            t_start=20.0,
            video_duration=600.0,
        )
        self.assertEqual(label.name, "substantive")


class SponsorDetection(unittest.TestCase):
    def test_3b1b_patreon_outro_at_end(self) -> None:
        label = classify_passage(
            "I want to give a special thanks to all of the supporters on Patreon "
            "who make this possible.",
            t_start=970.0,
            video_duration=WUVT_DURATION_SEC,
        )
        self.assertEqual(label.name, "sponsor")

    def test_sponsored_by_brand(self) -> None:
        label = classify_passage(
            "This video is sponsored by Brilliant. Brilliant is a learning platform.",
            t_start=300.0,
            video_duration=900.0,
        )
        self.assertEqual(label.name, "sponsor")

    def test_promo_code_with_url(self) -> None:
        label = classify_passage(
            "Head over to brilliant.org/3b1b and use code GRANT for 20% off.",
            t_start=320.0,
            video_duration=900.0,
        )
        self.assertEqual(label.name, "sponsor")

    def test_sponsor_position_independent(self) -> None:
        label_early = classify_passage(
            "This video is brought to you by Squarespace.",
            t_start=10.0,
            video_duration=600.0,
        )
        label_late = classify_passage(
            "This video is brought to you by Squarespace.",
            t_start=550.0,
            video_duration=600.0,
        )
        self.assertEqual(label_early.name, "sponsor")
        self.assertEqual(label_late.name, "sponsor")


class RecapAndTransition(unittest.TestCase):
    def test_recap_opener(self) -> None:
        label = classify_passage(
            "As we discussed earlier, the derivative of a sum is the sum of the derivatives."
        )
        self.assertEqual(label.name, "recap")

    def test_transition_opener(self) -> None:
        label = classify_passage(
            "Moving on, let's look at the chain rule."
        )
        self.assertEqual(label.name, "transition")

    def test_let_us_recap(self) -> None:
        label = classify_passage("Let's recap what we covered so far.")
        self.assertEqual(label.name, "recap")


class SubstantiveContent(unittest.TestCase):
    """The hard cases — these must NOT be flagged as structural."""

    def test_real_derivation_passage(self) -> None:
        label = classify_passage(
            "If you take the derivative of x squared, you get two x. "
            "This pattern generalizes — the derivative of x to the n is n times x to the n minus one.",
            t_start=300.0,
            video_duration=WUVT_DURATION_SEC,
        )
        self.assertEqual(label.name, "substantive")
        self.assertEqual(label.confidence, 0.0)

    def test_question_in_substantive_content(self) -> None:
        label = classify_passage(
            "Why does the area under a curve give you the integral? "
            "It comes down to a beautiful geometric fact.",
            t_start=400.0,
            video_duration=WUVT_DURATION_SEC,
        )
        self.assertEqual(label.name, "substantive")

    def test_empty_passage(self) -> None:
        self.assertEqual(classify_passage("").name, "substantive")

    def test_no_position_data_still_classifies(self) -> None:
        label = classify_passage("Hey everyone, welcome to the channel.")
        self.assertEqual(label.name, "intro")


class PenaltyTable(unittest.TestCase):
    def test_substantive_zero_penalty(self) -> None:
        label = classify_passage("If x is greater than y, then we have a problem.")
        self.assertEqual(label_penalty(label), 0.0)

    def test_sponsor_largest_penalty(self) -> None:
        sponsor = classify_passage(
            "This video is sponsored by Squarespace.",
            t_start=950.0,
            video_duration=1000.0,
        )
        intro = classify_passage(
            "Hey everyone, welcome back.",
            t_start=5.0,
            video_duration=1000.0,
        )
        self.assertGreater(label_penalty(sponsor), label_penalty(intro))

    def test_transition_smaller_than_intro(self) -> None:
        transition = classify_passage("Moving on, let's look at integrals.")
        intro = classify_passage(
            "Hey everyone, today we're going to talk about integrals.",
            t_start=5.0,
            video_duration=1000.0,
        )
        self.assertLess(label_penalty(transition), label_penalty(intro))


if __name__ == "__main__":
    unittest.main()
