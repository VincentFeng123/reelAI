import sys
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.knowledge_level import (  # noqa: E402
    KNOWLEDGE_LEVELS,
    LEVEL_VALUES,
    effective_level_target,
    normalize_knowledge_level,
)


class NormalizeTests(unittest.TestCase):
    def test_none_and_empty_default_to_beginner(self) -> None:
        self.assertEqual(normalize_knowledge_level(None), "beginner")
        self.assertEqual(normalize_knowledge_level("  "), "beginner")

    def test_case_and_whitespace_tolerant(self) -> None:
        self.assertEqual(normalize_knowledge_level(" Advanced "), "advanced")

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_knowledge_level("expert")


class TargetTests(unittest.TestCase):
    def test_mapping(self) -> None:
        self.assertEqual(KNOWLEDGE_LEVELS, ("beginner", "intermediate", "advanced"))
        self.assertEqual(LEVEL_VALUES["beginner"], 0.15)
        self.assertEqual(LEVEL_VALUES["intermediate"], 0.50)
        self.assertEqual(LEVEL_VALUES["advanced"], 0.85)

    def test_adjustment_applied_and_clamped(self) -> None:
        self.assertAlmostEqual(effective_level_target("beginner", 0.2), 0.35)
        # adjustment beyond the bound is clamped to ±0.35
        self.assertAlmostEqual(effective_level_target("beginner", 9.0), 0.50)
        self.assertAlmostEqual(effective_level_target("advanced", -9.0), 0.50)

    def test_result_clamped_to_unit_interval(self) -> None:
        self.assertAlmostEqual(effective_level_target("advanced", 0.35), 1.0)

    def test_none_inputs(self) -> None:
        self.assertAlmostEqual(effective_level_target(None, None), 0.15)


if __name__ == "__main__":
    unittest.main()
