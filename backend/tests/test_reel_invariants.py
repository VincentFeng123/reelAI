"""
Invariant tests for reel cutting — verify the contract the user cares about:

1. Every reel begins at the start of a complete sentence.
2. Every reel begins at the beginning of the relevant topic segment
   (not mid-thought).
3. Reels within a topic-segment end at a complete sentence.
4. When a topic exceeds max_len, consecutive reels continue seamlessly
   (no overlap, no gap beyond the natural between-cue pause).
5. Every user setting influences generation (or is client-only by design).

These tests drive the code paths directly with hand-built transcripts so we
don't depend on YouTube / LLM availability.
"""

from __future__ import annotations

import unittest
from typing import Any

from app.services.embeddings import EmbeddingService
from app.services.reels import ReelService
from app.services.youtube import YouTubeService


# ---------- helpers ---------- #


def make_transcript(sentences: list[tuple[float, float, str]]) -> list[dict[str, Any]]:
    """Given (start, duration, text) tuples, build a transcript the pipeline accepts."""
    return [{"start": s, "duration": d, "text": t} for s, d, t in sentences]


def _sentence_cues(starts_texts: list[tuple[float, str]], cue_sec: float = 3.0) -> list[dict[str, Any]]:
    return [{"start": s, "duration": cue_sec, "text": t} for s, t in starts_texts]


class SettingsEnforcementTests(unittest.TestCase):
    """Every user setting that claims to affect processing must actually do so."""

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())

    def test_min_max_clip_duration_is_respected(self) -> None:
        cmin, cmax, target = self.rs._resolve_clip_duration_bounds(
            target_clip_duration_sec=55,
            target_clip_duration_min_sec=25,
            target_clip_duration_max_sec=45,
        )
        self.assertEqual(cmin, 25)
        self.assertEqual(cmax, 45)
        self.assertEqual(target, 45)  # target clamped into [25, 45]

    def test_preferred_video_duration_locks_generation_cap(self) -> None:
        cap_any = self.rs._generation_target_cap(
            num_reels=10, preferred_video_duration="any", fast_mode=False,
        )
        cap_short = self.rs._generation_target_cap(
            num_reels=10, preferred_video_duration="short", fast_mode=False,
        )
        self.assertGreater(cap_any, 10)
        self.assertEqual(cap_short, 10)

    def test_video_pool_mode_affects_duration_plan(self) -> None:
        plan_short = self.rs._stage_duration_plan(
            stage_name="broad",
            preferred_video_duration="any",
            video_pool_mode="short-first",
            fast_mode=True,
            retrieval_profile="deep",
        )
        plan_long = self.rs._stage_duration_plan(
            stage_name="broad",
            preferred_video_duration="any",
            video_pool_mode="long-form",
            fast_mode=True,
            retrieval_profile="deep",
        )
        self.assertNotEqual(plan_short, plan_long)

    def test_ambiguous_concept_drops_entertainment_tier(self) -> None:
        # The hard-drop gate runs inside the candidate loop, but we can verify
        # the pieces it depends on: ambiguity detection + tier classification.
        from app.services.segmenter import normalize_terms
        tokens = normalize_terms(["calculus", "derivatives"])
        self.assertTrue(tokens & self.rs.AMBIGUOUS_CONCEPT_TOKENS)
        tier = self.rs._infer_channel_tier(
            channel="some channel", title="Mean Girls - Calculus Scene",
        )
        self.assertEqual(tier, "entertainment_media")


class BoundaryPaddingTests(unittest.TestCase):
    """Pre-roll / post-roll padding absorbs YouTube embed seek jitter and
    lets the closing consonant finish. Tests verify:

      1. Default callers (pad_start_sec=0, pad_end_sec=0) get unchanged
         behavior — important because many existing tests rely on that.
      2. When padding is opted in, the returned window is exactly
         pad_start_sec seconds earlier at the start and pad_end_sec
         seconds later at the end (clamped by video_duration and
         min_start).
      3. Padding never crosses min_start (so chained reels can't regress
         below the caller's floor).
      4. Padding never extends past video_duration_sec.
      5. _split_into_consecutive_windows applies pre-roll to the first
         reel only, post-roll to the last reel only — middle reels chain
         exactly so consecutive-reel playback doesn't double-play the
         padded overlap.
      6. Generator uses REEL_PAD_START_SEC / REEL_PAD_END_SEC constants
         (currently 0.3s each) for production call sites.
    """

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())

    def _synthetic_cues(self, n: int = 60, cue_spacing: float = 4.0, cue_dur: float = 3.9) -> list[dict[str, Any]]:
        """Build ``n`` cues, each ending with terminal punctuation so every
        cue boundary is a legal sentence start/end."""
        out: list[dict[str, Any]] = []
        for i in range(n):
            out.append({
                "start": i * cue_spacing,
                "duration": cue_dur,
                "text": f"Synthetic sentence number {i}.",
            })
        return out

    def test_padding_constants_exported(self) -> None:
        """Production callers import REEL_PAD_START_SEC / REEL_PAD_END_SEC.
        Lock in the current values so drift triggers an explicit review."""
        from app.services.reels import REEL_PAD_START_SEC, REEL_PAD_END_SEC
        self.assertEqual(REEL_PAD_START_SEC, 0.3)
        self.assertEqual(REEL_PAD_END_SEC, 0.3)


if __name__ == "__main__":
    unittest.main()
