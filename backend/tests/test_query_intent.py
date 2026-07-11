"""Golden table tests for ``backend/app/services/query_intent.py``.

The classifier must distinguish:
  * Broad surveys ("Physics 1 review", "AP Bio everything", "intro to calculus")
  * Narrow concepts ("torque", "B = qvBsinθ", "Treaty of Versailles")
  * Medium fallback (multi-token but no obvious signal)
  * None for empty input
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.query_intent import classify_query  # noqa: E402


class BroadQueries(unittest.TestCase):
    def test_bare_subject_physics(self) -> None:
        intent = classify_query("physics")
        self.assertEqual(intent.type, "broad")
        self.assertIn("physics", intent.anchors)

    def test_bare_subject_history(self) -> None:
        self.assertEqual(classify_query("history").type, "broad")

    def test_two_word_subject_phrase(self) -> None:
        self.assertEqual(classify_query("ap biology").type, "broad")

    def test_review_verb(self) -> None:
        self.assertEqual(classify_query("calculus review").type, "broad")

    def test_overview_verb(self) -> None:
        self.assertEqual(classify_query("Spanish 2 overview").type, "broad")

    def test_explain_verb(self) -> None:
        intent = classify_query("explain everything about derivatives")
        self.assertEqual(intent.type, "broad")

    def test_course_code(self) -> None:
        self.assertEqual(classify_query("PHYS 121").type, "broad")

    def test_ap_macro_econ(self) -> None:
        self.assertEqual(classify_query("AP Macroeconomics").type, "broad")


class NarrowQueries(unittest.TestCase):
    def test_single_concept_torque(self) -> None:
        intent = classify_query("torque")
        self.assertEqual(intent.type, "narrow")
        self.assertIn("torque", intent.anchors)

    def test_single_concept_photosynthesis(self) -> None:
        self.assertEqual(classify_query("photosynthesis").type, "narrow")

    def test_single_concept_federalism(self) -> None:
        self.assertEqual(classify_query("federalism").type, "narrow")

    def test_single_concept_osmosis(self) -> None:
        self.assertEqual(classify_query("osmosis").type, "narrow")

    def test_equation_query(self) -> None:
        intent = classify_query("F = ma")
        self.assertEqual(intent.type, "narrow")

    def test_equation_with_greek(self) -> None:
        self.assertEqual(classify_query("B = qvBsinθ").type, "narrow")

    def test_proper_noun_treaty(self) -> None:
        intent = classify_query("Treaty of Versailles")
        self.assertEqual(intent.type, "narrow")

    def test_proper_noun_dunning_kruger(self) -> None:
        self.assertEqual(classify_query("Dunning-Kruger effect").type, "narrow")

    def test_formula_pattern(self) -> None:
        intent = classify_query("the quadratic formula")
        self.assertEqual(intent.type, "narrow")

    def test_three_token_concept(self) -> None:
        self.assertEqual(classify_query("chain rule derivatives").type, "narrow")


class MediumAndNone(unittest.TestCase):
    def test_empty_string(self) -> None:
        intent = classify_query("")
        self.assertEqual(intent.type, "none")
        self.assertEqual(intent.anchors, ())

    def test_none_input(self) -> None:
        self.assertEqual(classify_query(None).type, "none")

    def test_whitespace_only(self) -> None:
        self.assertEqual(classify_query("   ").type, "none")

    def test_medium_fallback(self) -> None:
        intent = classify_query("how do plants make sugar from light")
        self.assertEqual(intent.type, "medium")
        self.assertGreater(len(intent.anchors), 0)


class AnchorExtraction(unittest.TestCase):
    def test_proper_noun_kept_as_anchor(self) -> None:
        intent = classify_query("Treaty of Versailles consequences")
        self.assertTrue(any("Treaty of Versailles" in a for a in intent.anchors))

    def test_equation_kept_as_anchor(self) -> None:
        intent = classify_query("derive F = ma")
        self.assertTrue(any("F" in a and "=" in a for a in intent.anchors))

    def test_lowercase_token_anchors(self) -> None:
        intent = classify_query("torque")
        self.assertEqual(intent.anchors, ("torque",))


if __name__ == "__main__":
    unittest.main()
