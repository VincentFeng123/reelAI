"""Targeted tests for ``importance_ranker._topic_concentration``.

The function must distinguish:
  * A segment that **derives** an anchor (long mention window + discourse markers)
    from one that just **name-drops** it.
  * A segment whose anchor is **rare** (high IDF) from one whose anchor is
    **everywhere** (low IDF — should not concentrate on a casual mention).
  * Lemma/token matching: ``torque`` matches ``torques`` / ``torquing`` but
    ``force`` is NOT matched inside ``enforce`` / ``forced``.
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

from backend.app.services.importance_ranker import _topic_concentration  # noqa: E402
from backend.app.services.segment_features import (  # noqa: E402
    SegmentFeatures,
    extract_features,
    match_anchors_in_text,
)
from backend.app.services.segmenter import SegmentMatch  # noqa: E402


def _seg(chunk_index: int, t_start: float, t_end: float, text: str) -> SegmentMatch:
    return SegmentMatch(
        chunk_index=chunk_index, t_start=t_start, t_end=t_end, text=text, score=0.5,
    )


def _features_for(seg: SegmentMatch, *, others: list[SegmentMatch] | None = None) -> tuple[SegmentFeatures, dict[str, float]]:
    """Run the real extractor against [seg, *others] so we get realistic IDF."""
    bundle = extract_features(
        [seg] + (others or []),
        embedder=None, nlp=None, video_duration=300.0, conn=None,
    )
    return bundle.segments[0], bundle.idf


class DerivedVsNameDropped(unittest.TestCase):
    def test_derived_outranks_name_dropped(self) -> None:
        derived = _seg(
            0, 0.0, 60.0,
            "Torque is the rotational analog of force. Notice that torque depends on the lever arm. "
            "We call this the moment arm. Because the lever arm matters, longer wrenches produce more torque. "
            "Specifically, torque equals force times the perpendicular distance. "
            "For example, applying force at the end of the wrench gives the most torque.",
        )
        name_dropped = _seg(
            0, 0.0, 60.0,
            "Today we'll cover a lot of physics topics. Torque is one. Mass is another. "
            "Energy comes up too. Torque again is interesting. So is gravity.",
        )
        derived_feats, derived_idf = _features_for(
            derived,
            others=[
                _seg(1, 60.0, 90.0, "Other physics content here about velocity and mass."),
                _seg(2, 90.0, 120.0, "More about kinetic energy and acceleration."),
            ],
        )
        nd_feats, nd_idf = _features_for(
            name_dropped,
            others=[
                _seg(1, 60.0, 90.0, "Other physics content here about velocity and mass."),
                _seg(2, 90.0, 120.0, "More about kinetic energy and acceleration."),
            ],
        )
        derived_score = _topic_concentration(
            derived, derived_feats, ("torque",), derived_idf, nlp=None,
        )
        nd_score = _topic_concentration(
            name_dropped, nd_feats, ("torque",), nd_idf, nlp=None,
        )
        self.assertGreater(derived_score, nd_score)
        self.assertLess(nd_score, 0.5)


class IdfDownweighting(unittest.TestCase):
    """A casual mention of a low-IDF anchor should NOT concentrate."""

    def test_low_idf_anchor_alone_does_not_qualify(self) -> None:
        # 12 segments: "force" appears in 11 (max_df=0.8 drops it from vocab →
        # idf.get fallback = 1.0). "centripetal" appears in only 1 → high IDF.
        force_segs = [
            _seg(i, i * 30.0, (i + 1) * 30.0,
                 f"Force is the topic of segment {i}. Force comes up here too.")
            for i in range(11)
        ]
        rare_seg = _seg(11, 330.0, 360.0,
                         "Centripetal acceleration arises in circular motion paths.")
        bundle = extract_features(
            force_segs + [rare_seg], embedder=None, nlp=None,
            video_duration=400.0, conn=None,
        )
        target_seg = force_segs[0]
        target_feats = bundle.segments[0]
        score = _topic_concentration(
            target_seg, target_feats, ("force", "centripetal"),
            bundle.idf, nlp=None,
        )
        # idf("centripetal") ≫ idf("force") fallback (1.0), so a force-only
        # sentence cannot clear 0.5 × max_w → no qualifying mentions → 0.
        self.assertEqual(score, 0.0)

    def test_high_idf_anchor_qualifies(self) -> None:
        # Co-mention scenario: same vocab as above but the target segment is
        # the rare-anchor segment. Centripetal mention DOES qualify because
        # the per-sentence weight clears the threshold.
        force_segs = [
            _seg(i, i * 30.0, (i + 1) * 30.0,
                 f"Force is the topic of segment {i}. Force comes up here too.")
            for i in range(11)
        ]
        rare_seg = _seg(11, 330.0, 360.0,
                         "Centripetal acceleration arises in circular motion paths. "
                         "Notice that centripetal force points inward toward the center.")
        bundle = extract_features(
            force_segs + [rare_seg], embedder=None, nlp=None,
            video_duration=400.0, conn=None,
        )
        target_feats = bundle.segments[-1]  # rare_seg's features
        score = _topic_concentration(
            rare_seg, target_feats, ("force", "centripetal"),
            bundle.idf, nlp=None,
        )
        self.assertGreater(score, 0.0)


class LemmaMatching(unittest.TestCase):
    def test_torque_matches_inflections(self) -> None:
        hits = match_anchors_in_text(
            "She studied torques in lab and the torquing motion.",
            anchors=["torque"],
        )
        self.assertGreaterEqual(hits.get("torque", 0), 1)

    def test_force_does_not_match_inside_enforce(self) -> None:
        # ``enforce`` is a different lemma from ``force`` — must not match.
        # (We don't include ``forced`` here because that IS a legitimate
        # past-tense inflection of ``force`` and SHOULD match.)
        hits = match_anchors_in_text(
            "We must enforce the law in every district.",
            anchors=["force"],
        )
        self.assertEqual(hits.get("force", 0), 0)

    def test_multi_word_anchor_ngram_match(self) -> None:
        hits = match_anchors_in_text(
            "The Treaty of Versailles ended the war.",
            anchors=["Treaty of Versailles"],
        )
        self.assertEqual(hits.get("Treaty of Versailles", 0), 1)


class EmptyEdgeCases(unittest.TestCase):
    def test_no_anchors_returns_zero(self) -> None:
        seg = _seg(0, 0.0, 30.0, "Some text here.")
        feats, idf = _features_for(seg)
        self.assertEqual(_topic_concentration(seg, feats, (), idf, nlp=None), 0.0)

    def test_empty_text_returns_zero(self) -> None:
        seg = _seg(0, 0.0, 30.0, "")
        feats, idf = _features_for(seg)
        self.assertEqual(_topic_concentration(seg, feats, ("torque",), idf, nlp=None), 0.0)


if __name__ == "__main__":
    unittest.main()
