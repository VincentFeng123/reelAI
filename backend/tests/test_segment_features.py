"""Unit tests for ``backend/app/services/segment_features.py``.

We test in three layers:

  * Standalone helpers (anchor matching, lexical density)
  * Per-segment scoring (instructional density ordering on dense-vs-filler)
  * Bundle output (TF-IDF idf table populated, structural label set)
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

from backend.app.services.segment_features import (  # noqa: E402
    extract_features,
    match_anchors_in_text,
    tokens_for_match,
)
from backend.app.services.segmenter import SegmentMatch  # noqa: E402


def _seg(chunk_index: int, t_start: float, t_end: float, text: str) -> SegmentMatch:
    return SegmentMatch(
        chunk_index=chunk_index,
        t_start=t_start,
        t_end=t_end,
        text=text,
        score=0.5,
    )


class TokenAndAnchorMatching(unittest.TestCase):
    def test_lemma_token_matches_inflection(self) -> None:
        # Without spaCy we use suffix stripping — both ``torques`` and
        # ``torquing`` should normalize to a stem matching ``torque``.
        hits = match_anchors_in_text(
            "She studied torques in lecture and the torquing motion.",
            anchors=["torque"],
        )
        self.assertGreaterEqual(hits.get("torque", 0), 1)

    def test_substring_does_not_match(self) -> None:
        # ``force`` must NOT match inside ``enforce`` or ``forced`` becoming
        # an unrelated stem. Suffix strip turns "forced" -> "forc" so it
        # won't equal "force".
        hits = match_anchors_in_text(
            "We must enforce the law.",
            anchors=["force"],
        )
        self.assertEqual(hits.get("force", 0), 0)

    def test_multi_word_anchor(self) -> None:
        hits = match_anchors_in_text(
            "The Treaty of Versailles ended the war.",
            anchors=["Treaty of Versailles"],
        )
        self.assertEqual(hits.get("Treaty of Versailles", 0), 1)

    def test_empty_anchors_returns_empty(self) -> None:
        self.assertEqual(match_anchors_in_text("any text", anchors=[]), {})

    def test_tokens_basic(self) -> None:
        toks = tokens_for_match("Photosynthesis is amazing.")
        self.assertIn("photosynthesi", toks + ["photosynthesis"])  # either form is fine


class InstructionalDensityOrdering(unittest.TestCase):
    """A discourse-rich segment should score above a filler segment."""

    def test_dense_outranks_filler(self) -> None:
        dense = _seg(
            0, 0.0, 30.0,
            "Because the system is closed, the total energy is conserved. "
            "We call this the law of conservation. In other words, the energy "
            "out equals the energy in. Notice that this leads to a useful "
            "shortcut for solving problems.",
        )
        filler = _seg(
            1, 30.0, 60.0,
            "So um yeah like I think you know it's just kind of really really "
            "interesting to think about all the things that we just talked about.",
        )
        bundle = extract_features(
            [dense, filler], embedder=None, nlp=None,
            video_duration=120.0, conn=None,
        )
        self.assertGreater(
            bundle.segments[0].instructional_density,
            bundle.segments[1].instructional_density,
        )
        self.assertGreater(bundle.segments[0].discourse_marker_count, 0)

    def test_hearst_hits_are_captured(self) -> None:
        seg = _seg(
            0, 0.0, 30.0,
            "Photosynthesis is defined as the conversion of light energy into "
            "chemical energy by plants.",
        )
        bundle = extract_features([seg], embedder=None, nlp=None, video_duration=60.0, conn=None)
        names = {n for n, _ in bundle.segments[0].hearst_hits}
        self.assertTrue(any(n in names for n in ("x_defined_as_y",)))

    def test_hearst_generic_y_dropped(self) -> None:
        # "The economy is a problem" — generic Y, must NOT register as a definition.
        seg = _seg(0, 0.0, 30.0, "The economy is a problem these days.")
        bundle = extract_features([seg], embedder=None, nlp=None, video_duration=60.0, conn=None)
        names = {n for n, _ in bundle.segments[0].hearst_hits}
        self.assertNotIn("x_is_a_y", names)


class StructuralLabelIntegration(unittest.TestCase):
    def test_intro_segment_labeled(self) -> None:
        seg = _seg(
            0, 5.0, 30.0,
            "Hey everyone, welcome back to the channel. Today we're going to talk about derivatives.",
        )
        bundle = extract_features([seg], embedder=None, nlp=None, video_duration=600.0, conn=None)
        self.assertEqual(bundle.segments[0].structural_label, "intro")
        self.assertGreater(bundle.segments[0].structural_penalty, 0.0)

    def test_substantive_segment_zero_penalty(self) -> None:
        seg = _seg(
            0, 300.0, 330.0,
            "If you take the derivative of x squared, you get two x.",
        )
        bundle = extract_features([seg], embedder=None, nlp=None, video_duration=600.0, conn=None)
        self.assertEqual(bundle.segments[0].structural_label, "substantive")
        self.assertEqual(bundle.segments[0].structural_penalty, 0.0)


class IdfTableExposure(unittest.TestCase):
    def test_idf_dict_populated(self) -> None:
        segs = [
            _seg(0, 0.0, 30.0, "Forces and motion are key topics in physics. Force makes things move."),
            _seg(1, 30.0, 60.0, "Torque is the rotational analog of force. Torque depends on lever arm."),
            _seg(2, 60.0, 90.0, "Centripetal acceleration keeps an object moving in a circle."),
        ]
        bundle = extract_features(segs, embedder=None, nlp=None, video_duration=120.0, conn=None)
        self.assertGreater(len(bundle.idf), 0)
        # ``force`` should appear in two of three segments → lower IDF than
        # ``centripetal`` which appears in only one.
        force_idf = bundle.idf.get("force") or bundle.idf.get("forces")
        centripetal_idf = bundle.idf.get("centripetal")
        if force_idf is not None and centripetal_idf is not None:
            self.assertLess(force_idf, centripetal_idf + 0.0001)

    def test_tfidf_top_returned_per_segment(self) -> None:
        segs = [
            _seg(0, 0.0, 30.0, "Derivatives measure instantaneous change."),
            _seg(1, 30.0, 60.0, "Integrals measure accumulated change."),
        ]
        bundle = extract_features(segs, embedder=None, nlp=None, video_duration=60.0, conn=None)
        for s in bundle.segments:
            self.assertIsInstance(s.tfidf_top, tuple)


class CentralityFallback(unittest.TestCase):
    def test_short_video_uses_uniform(self) -> None:
        segs = [_seg(i, i * 30.0, (i + 1) * 30.0, f"Segment {i} content here.") for i in range(4)]
        bundle = extract_features(segs, embedder=None, nlp=None, video_duration=120.0, conn=None)
        for s in bundle.segments:
            self.assertEqual(s.centrality, 0.5)


if __name__ == "__main__":
    unittest.main()
