import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import _resolve_target_clip_duration_bounds
from backend.app.services.material_intelligence import MaterialIntelligenceService
from backend.app.services.reels import ReelService


class MediumRegressionTests(unittest.TestCase):
    def test_main_clip_bounds_respect_minimum_duration(self) -> None:
        self.assertEqual(
            _resolve_target_clip_duration_bounds(15, None, None),
            (15, 15, 30),
        )

    def test_reel_service_clip_bounds_respect_minimum_duration(self) -> None:
        service = ReelService(embedding_service=None, youtube_service=None)
        self.assertEqual(
            service._resolve_clip_duration_bounds(15, None, None),
            (15, 30, 15),
        )

    def test_material_intelligence_cache_key_uses_full_text(self) -> None:
        service = MaterialIntelligenceService()
        shared_prefix = "a" * 24_000
        key_a = service._cache_key(shared_prefix + "first-suffix", "systems", 12)
        key_b = service._cache_key(shared_prefix + "second-suffix", "systems", 12)
        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
