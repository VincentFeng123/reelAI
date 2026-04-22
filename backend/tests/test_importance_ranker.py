"""Unit tests for ``backend/app/services/importance_ranker.py``.

Covers:
  * ``select_clips`` honors intent target_n (narrow=≤2, broad=3-5)
  * Sponsor / outro segments are filtered out, never selected
  * Cluster-diversified selection prefers distinct cluster ids
  * Temporal overlap suppression rejects near-duplicate windows
  * Cross-encoder gate fires on equation / single-concept / multi-word PN queries
  * Importance weighting respects intent type (narrow vs broad weight tables)
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

from backend.app.services.importance_ranker import (  # noqa: E402
    RankedSegment,
    _should_rerank,
    _temporal_overlap_ratio,
    rank_segments,
    select_clips,
)
from backend.app.services.query_intent import QueryIntent, classify_query  # noqa: E402
from backend.app.services.segment_features import extract_features  # noqa: E402
from backend.app.services.segmenter import SegmentMatch  # noqa: E402


def _seg(chunk_index: int, t_start: float, t_end: float, text: str) -> SegmentMatch:
    return SegmentMatch(
        chunk_index=chunk_index,
        t_start=t_start,
        t_end=t_end,
        text=text,
        score=0.5,
    )


class SelectClipsCount(unittest.TestCase):
    def _build(self, n: int) -> list[SegmentMatch]:
        return [
            _seg(i, i * 30.0, (i + 1) * 30.0,
                 f"Segment {i} explains topic {i}. Because we know the rule, we apply it.")
            for i in range(n)
        ]

    def test_narrow_returns_at_most_two(self) -> None:
        segs = self._build(8)
        bundle = extract_features(segs, embedder=None, nlp=None,
                                   video_duration=240.0, conn=None)
        intent = QueryIntent(type="narrow", confidence=0.9, anchors=("topic",),
                             raw_query="topic", normalized_query="topic")
        ranked = rank_segments(segs, bundle, intent, embedder=None, conn=None)
        picks = select_clips(ranked, intent)
        self.assertLessEqual(len(picks), 2)
        self.assertGreaterEqual(len(picks), 1)

    def test_broad_returns_three_to_five(self) -> None:
        segs = self._build(8)
        bundle = extract_features(segs, embedder=None, nlp=None,
                                   video_duration=240.0, conn=None)
        intent = QueryIntent(type="broad", confidence=0.9, anchors=("topic",),
                             raw_query="topic review", normalized_query="topic review")
        ranked = rank_segments(segs, bundle, intent, embedder=None, conn=None)
        picks = select_clips(ranked, intent)
        self.assertGreaterEqual(len(picks), 3)
        self.assertLessEqual(len(picks), 5)


class StructuralFiltering(unittest.TestCase):
    def test_sponsor_segment_never_selected(self) -> None:
        substantive = _seg(
            0, 60.0, 90.0,
            "If you take the derivative of x squared, you get two x. "
            "Because we apply the power rule, we drop the exponent.",
        )
        sponsor = _seg(
            1, 90.0, 120.0,
            "This video is sponsored by Brilliant. Brilliant is a learning platform.",
        )
        substantive2 = _seg(
            2, 120.0, 150.0,
            "Notice that the chain rule is what we call a composition rule.",
        )
        bundle = extract_features(
            [substantive, sponsor, substantive2], embedder=None, nlp=None,
            video_duration=600.0, conn=None,
        )
        intent = QueryIntent(type="medium", confidence=0.5, anchors=("derivative",),
                             raw_query="derivative", normalized_query="derivative")
        ranked = rank_segments([substantive, sponsor, substantive2], bundle, intent,
                                embedder=None, conn=None)
        picks = select_clips(ranked, intent, target_n=3)
        for p in picks:
            self.assertNotEqual(p.features.structural_label, "sponsor")

    def test_intro_and_recap_segments_never_selected(self) -> None:
        intro = _seg(
            0, 0.0, 30.0,
            "Hey everyone welcome back to the channel, today we are diving in.",
        )
        recap = _seg(
            1, 30.0, 60.0,
            "Let's recap what we covered so far before moving on.",
        )
        substantive = _seg(
            2, 60.0, 90.0,
            "The derivative measures instantaneous rate of change at a point.",
        )
        bundle = extract_features(
            [intro, recap, substantive], embedder=None, nlp=None,
            video_duration=600.0, conn=None,
        )
        intent = QueryIntent(type="medium", confidence=0.5, anchors=("derivative",),
                             raw_query="derivative", normalized_query="derivative")
        ranked = rank_segments([intro, recap, substantive], bundle, intent,
                                embedder=None, conn=None)
        picks = select_clips(ranked, intent, target_n=3)
        labels = {p.features.structural_label for p in picks}
        self.assertNotIn("intro", labels)
        self.assertNotIn("recap", labels)


class ClusterDiversification(unittest.TestCase):
    def test_distinct_clusters_preferred(self) -> None:
        # Three groups of two segments — same TF-IDF top term within group.
        segs = [
            _seg(0, 0, 30, "Forces and motion explain how things move. Force is fundamental."),
            _seg(1, 30, 60, "Force determines acceleration. Newton's law links force to mass."),
            _seg(2, 60, 90, "Photosynthesis converts light into chemical energy. Plants use photosynthesis."),
            _seg(3, 90, 120, "Photosynthesis happens in chloroplasts. Photosynthesis is essential."),
            _seg(4, 120, 150, "Derivatives measure instantaneous change. The derivative of x^2 is 2x."),
            _seg(5, 150, 180, "Derivatives generalize to higher dimensions. Derivative rules apply."),
        ]
        bundle = extract_features(segs, embedder=None, nlp=None,
                                   video_duration=200.0, conn=None)
        intent = QueryIntent(type="broad", confidence=0.9, anchors=(),
                             raw_query="science overview",
                             normalized_query="science overview")
        ranked = rank_segments(segs, bundle, intent, embedder=None, conn=None)
        picks = select_clips(ranked, intent)
        # At least 3 distinct cluster ids in the broad pick.
        cluster_ids = {p.cluster_id for p in picks}
        self.assertGreaterEqual(len(cluster_ids), 3)


class TemporalOverlap(unittest.TestCase):
    def test_overlap_ratio_full(self) -> None:
        a = _seg(0, 0.0, 30.0, "x")
        b = _seg(1, 0.0, 30.0, "y")
        self.assertAlmostEqual(_temporal_overlap_ratio(a, b), 1.0)

    def test_overlap_ratio_disjoint(self) -> None:
        a = _seg(0, 0.0, 30.0, "x")
        b = _seg(1, 60.0, 90.0, "y")
        self.assertEqual(_temporal_overlap_ratio(a, b), 0.0)

    def test_overlap_ratio_partial(self) -> None:
        a = _seg(0, 0.0, 30.0, "x")
        b = _seg(1, 20.0, 50.0, "y")
        # Overlap = 10s, shorter = 30s → 1/3.
        self.assertAlmostEqual(_temporal_overlap_ratio(a, b), 10.0 / 30.0, places=3)


class CrossEncoderGate(unittest.TestCase):
    def test_equation_query_triggers_rerank(self) -> None:
        intent = classify_query("F = ma")
        self.assertTrue(_should_rerank(intent, [0.7] * 10))

    def test_single_concept_triggers_rerank(self) -> None:
        intent = classify_query("torque")
        self.assertTrue(_should_rerank(intent, [0.7] * 10))

    def test_multiword_proper_noun_triggers_rerank(self) -> None:
        intent = classify_query("Treaty of Versailles")
        self.assertTrue(_should_rerank(intent, [0.7] * 10))

    def test_school_subject_does_not_trigger_solo(self) -> None:
        intent = QueryIntent(type="narrow", confidence=0.5, anchors=("physics",),
                             raw_query="physics", normalized_query="physics")
        # School subject + clearly separated bi scores → no rerank.
        self.assertFalse(_should_rerank(intent, [0.9, 0.5, 0.4, 0.3, 0.2]))

    def test_ambiguous_bi_scores_trigger_rerank(self) -> None:
        intent = QueryIntent(type="narrow", confidence=0.5, anchors=("physics",),
                             raw_query="physics", normalized_query="physics")
        self.assertTrue(_should_rerank(intent, [0.81, 0.80, 0.79, 0.78, 0.77]))


class IntentWeighting(unittest.TestCase):
    """Narrow intent should weight query relevance and topic concentration
    more heavily than broad intent does."""

    def test_narrow_weight_table_emphasizes_query_and_topic(self) -> None:
        from backend.app.services.importance_ranker import _WEIGHTS_BY_INTENT
        narrow = _WEIGHTS_BY_INTENT["narrow"]
        broad = _WEIGHTS_BY_INTENT["broad"]
        self.assertGreater(narrow["wq"], broad["wq"])
        self.assertGreater(narrow["wt"], broad["wt"])
        self.assertLess(narrow["wc"], broad["wc"])


class EmptyInput(unittest.TestCase):
    def test_empty_segments_returns_empty(self) -> None:
        intent = QueryIntent(type="none", confidence=1.0, anchors=(),
                             raw_query="", normalized_query="")
        from backend.app.services.segment_features import VideoFeatureBundle
        ranked = rank_segments([], VideoFeatureBundle(segments=(), idf={}), intent)
        self.assertEqual(ranked, [])
        self.assertEqual(select_clips(ranked, intent), [])


if __name__ == "__main__":
    unittest.main()
