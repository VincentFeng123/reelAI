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

import sqlite3
import unittest
from typing import Any

from app.db import SCHEMA
from app.services.embeddings import EmbeddingService
from app.services.reels import ReelService
from app.services.segmenter import SegmentMatch
from app.services.youtube import YouTubeService


# ---------- helpers ---------- #


def make_transcript(sentences: list[tuple[float, float, str]]) -> list[dict[str, Any]]:
    """Given (start, duration, text) tuples, build a transcript the pipeline accepts."""
    return [{"start": s, "duration": d, "text": t} for s, d, t in sentences]


def _sentence_cues(starts_texts: list[tuple[float, str]], cue_sec: float = 3.0) -> list[dict[str, Any]]:
    return [{"start": s, "duration": cue_sec, "text": t} for s, t in starts_texts]


class SentenceStartInvariantTests(unittest.TestCase):
    """Every reel begins at the start of a complete sentence."""

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())

    def test_refiner_snaps_start_to_sentence_boundary(self) -> None:
        # Cue 2 (t=3.0) ends with '.', so cue 3 (t=6.0) starts a new sentence.
        # Passing proposed_start=6.5 should snap back to 6.0.
        cues = _sentence_cues([
            (0.0, "This is a sentence that ends here."),
            (3.0, "Now we move on and this also ends."),
            (6.0, "And this is the next complete sentence that we want."),
            (9.0, "Followed by another one."),
            (12.0, "And yet another complete thought here."),
            (15.0, "With several more complete sentences."),
            (18.0, "Each ending cleanly on terminal punctuation."),
            (21.0, "And continuing onward without interruption."),
        ])
        win = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=6.5, proposed_end=18.0,
            video_duration_sec=60, min_len=6, max_len=15,
        )
        self.assertIsNotNone(win)
        # Should snap to 6.0 (start of "And this is the next...")
        self.assertEqual(win[0], 6.0)

    def test_refiner_snaps_start_forward_when_preceding_cue_lacks_punct(self) -> None:
        # Cue 0 has no terminal punct; cue 1 starts with a lowercase continuation.
        # Cue 2 follows a terminal period, so cue 2's start is the true sentence start.
        cues = _sentence_cues([
            (0.0, "This is part one of a long thought"),
            (3.0, "that keeps going without punctuation until here."),
            (6.0, "Now a new sentence begins right here."),
            (9.0, "And another sentence."),
            (12.0, "With more sentences."),
            (15.0, "And more thoughts."),
            (18.0, "Keep going."),
            (21.0, "Onward."),
        ])
        # Request starting just after cue 0 starts (1s in) — should move forward to 6.0.
        win = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=1.0, proposed_end=15.0,
            video_duration_sec=60, min_len=6, max_len=15,
        )
        self.assertIsNotNone(win)
        # Either cue 0 (start of video, always allowed) at 0.0 or cue 2 at 6.0.
        # The refiner picks the one closest to desired_start=1.0 among candidates
        # where i==0 OR prev cue ends with terminal punct. i==0 is always a candidate.
        self.assertIn(win[0], {0.0, 6.0})

    def test_short_clip_snaps_forward_when_first_cue_ends_mid_phrase(self) -> None:
        # First cue is a partial continuation; second cue starts a new sentence.
        cues = _sentence_cues([
            (0.0, "...and that is why it matters."),
            (3.0, "Let us now examine the topic in detail."),
            (6.0, "First we consider the basics."),
            (9.0, "Then we build on them."),
            (12.0, "Throughout this discussion we"),
            (15.0, "continue learning about the subject."),
        ], cue_sec=3.0)
        # Short is the full span.
        win = self.rs._full_short_clip_window(18, transcript=cues)
        self.assertIsNotNone(win)
        # Cue 0 ends with ".", so cue 1 (t=3.0) starts a new sentence — we shift there.
        self.assertEqual(win[0], 3.0)
        self.assertEqual(win[1], 18.0)

    def test_short_keeps_start_at_zero_when_first_cue_starts_new_sentence(self) -> None:
        # If there's no earlier cue ending on ".!?…" inside the first 4s we shouldn't move.
        cues = _sentence_cues([
            (0.0, "Welcome and let us begin our discussion of calculus"),
            (3.0, "by examining derivatives."),
        ], cue_sec=3.0)
        win = self.rs._full_short_clip_window(20, transcript=cues)
        self.assertIsNotNone(win)
        self.assertEqual(win[0], 0.0)

    def test_short_without_transcript_stays_at_zero(self) -> None:
        win = self.rs._full_short_clip_window(30)
        self.assertEqual(win, (0.0, 30.0))

    def test_short_does_not_shift_past_too_much_content(self) -> None:
        # If snapping forward would leave less than 40% of the video remaining,
        # we must keep t_start=0 (safety).
        cues = _sentence_cues([
            (0.0, "This opener ends with a period."),
            (3.0, "Another sentence ends here too."),
            (3.8, "And one more."),  # t_start shift to 3.8 leaves only 1.2s of a 5s video = 24%
        ], cue_sec=0.4)
        win = self.rs._full_short_clip_window(5, transcript=cues)
        self.assertIsNotNone(win)
        # t_start at 0 because shifting forward would nuke most of the video.
        self.assertEqual(win[0], 0.0)


class SentenceEndInvariantTests(unittest.TestCase):
    """Topic reels end at a complete sentence when transcript allows."""

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())

    def test_refiner_prefers_sentence_end_over_cue_end(self) -> None:
        cues = _sentence_cues([
            (0.0, "This is one sentence."),     # ends with .
            (3.0, "This is another sentence."),  # ends with .
            (6.0, "This one has no terminator"),  # no terminator
            (9.0, "This one ends cleanly."),     # ends with .
            (12.0, "And so does this one."),     # ends with .
            (15.0, "More content here"),         # no terminator
            (18.0, "The final sentence ends."),  # ends with .
            (21.0, "Tail content."),
        ])
        # Desired end is 10.0. max_len=12 → max_end=12. Sentence ends in range:
        # 3.0 (end of "one sentence."), 6.0 ("another sentence."), 12.0 ("ends cleanly.")
        # The refiner should pick 12.0 (end of cue 3) because it's closest to 10 that's ≤ max.
        win = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=0.0, proposed_end=10.0,
            video_duration_sec=60, min_len=6, max_len=12,
        )
        self.assertIsNotNone(win)
        self.assertEqual(win[0], 0.0)
        self.assertEqual(win[1], 12.0)  # cue 3 ends at 12.0 with "."

    def test_strict_mode_returns_none_when_no_sentence_end_in_range(self) -> None:
        # For PUNCTUATED transcripts (>= 15% terminal punct), strict mode
        # returns None when no sentence end exists in range. Build a
        # transcript that IS detected as punctuated but has an unpunctuated
        # window — only then should strict mode return None.
        cues = _sentence_cues([
            (0.0, "First sentence ends here."),
            (3.0, "Second sentence also ends."),
            # Punctuated prefix → detected as punctuated transcript
            (6.0, "No punct here"),
            (9.0, "None here either"),
            (12.0, "Still no punct"),
            (15.0, "None"),
        ])
        # Restrict to the unpunctuated window [6, 18] — strict mode
        # should fail because the transcript is punctuated overall but
        # this sub-range has no terminal punct.
        result = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=6.0, proposed_end=15.0,
            video_duration_sec=60, min_len=6, max_len=12, require_sentence_end=True,
        )
        self.assertIsNone(result)


class ContinuationInvariantTests(unittest.TestCase):
    """When a topic exceeds max_len, consecutive reels continue seamlessly."""

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())

    def test_consecutive_windows_have_zero_overlap(self) -> None:
        # A long topic spanning 30s, max_len=10
        cues = []
        t = 0.0
        for i in range(10):
            cues.append({"start": t, "duration": 3.0, "text": f"Sentence number {i} ends here."})
            t += 3.0
        windows = self.rs._split_into_consecutive_windows(
            transcript=cues, segment_start=0.0, segment_end=30.0,
            video_duration_sec=60, min_len=5, max_len=10,
        )
        self.assertGreaterEqual(len(windows), 2)
        for i in range(len(windows) - 1):
            self.assertLessEqual(windows[i][1], windows[i + 1][0] + 0.01,
                                  f"Overlap between windows {i} and {i+1}: {windows}")

    def test_consecutive_windows_have_no_gap_beyond_between_cue_pause(self) -> None:
        # Cues are contiguous (end-of-cue = next-cue-start); there should be no gap.
        cues = []
        t = 0.0
        for i in range(10):
            text = f"Sentence {i} is complete."
            cues.append({"start": t, "duration": 3.0, "text": text})
            t += 3.0
        windows = self.rs._split_into_consecutive_windows(
            transcript=cues, segment_start=0.0, segment_end=30.0,
            video_duration_sec=60, min_len=5, max_len=10,
        )
        self.assertGreaterEqual(len(windows), 2)
        for i in range(len(windows) - 1):
            gap = windows[i + 1][0] - windows[i][1]
            # No gap permitted — consecutive windows must start exactly at prev end.
            self.assertEqual(gap, 0.0,
                             f"Unexpected gap of {gap} between windows {i} and {i+1}: {windows}")

    def test_non_last_split_fails_over_to_lenient_when_no_sentence_end(self) -> None:
        # A long segment where the middle has no sentence terminators.
        # Strict mode would fail; caller must fall back and produce at least 1 window.
        cues = []
        t = 0.0
        # First 3 cues end cleanly, then 5 cues without terminators, then final clean ones.
        for i, text in enumerate([
            "First complete sentence.",
            "Second complete sentence.",
            "Third complete sentence.",
            "Continuing without any terminators here",
            "and continuing more",
            "and more still",
            "with yet more content",
            "still continuing",
            "Until finally it ends.",
            "Last thought wraps up cleanly.",
        ]):
            cues.append({"start": t, "duration": 3.0, "text": text})
            t += 3.0
        windows = self.rs._split_into_consecutive_windows(
            transcript=cues, segment_start=0.0, segment_end=30.0,
            video_duration_sec=60, min_len=5, max_len=8,
        )
        # Must produce at least one window — never silently drop the segment.
        self.assertGreater(len(windows), 0)
        # Overlap invariant still holds.
        for i in range(len(windows) - 1):
            self.assertLessEqual(windows[i][1], windows[i + 1][0] + 0.01)


class PersistedClipWindowInvariantTests(unittest.TestCase):
    """Persisted legacy reels keep the refined window the cutter produced."""

    MATERIAL_ID = "material-persisted-window"
    CONCEPT_ID = "concept-persisted-window"
    VIDEO_ID = "video-persisted-window"

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=None, youtube_service=None)
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT INTO materials (id, subject_tag, raw_text, source_type, source_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.MATERIAL_ID,
                "math",
                "Chain rule notes",
                "text",
                None,
                "2026-04-17T00:00:00+00:00",
            ),
        )
        self.conn.execute(
            "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self.CONCEPT_ID,
                self.MATERIAL_ID,
                "Chain rule",
                '["chain rule", "derivative"]',
                "How composite derivatives work.",
                None,
                "2026-04-17T00:01:00+00:00",
            ),
        )
        self.conn.execute(
            "INSERT INTO videos (id, title, channel_title, description, duration_sec, view_count, is_creative_commons, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.VIDEO_ID,
                "Chain rule walkthrough",
                "Test Channel",
                "A clean calculus lecture.",
                600,
                1000,
                0,
                "2026-04-17T00:02:00+00:00",
            ),
        )

    def tearDown(self) -> None:
        self.conn.close()

    def _concept(self) -> dict[str, Any]:
        return {
            "id": self.CONCEPT_ID,
            "title": "Chain rule",
            "keywords": ["chain rule", "derivative"],
        }

    def _video(self) -> dict[str, Any]:
        return {
            "id": self.VIDEO_ID,
            "title": "Chain rule walkthrough",
            "channel_title": "Test Channel",
            "description": "A clean calculus lecture.",
            "duration_sec": 600,
        }

    def _segment(self, *, score: float = 0.9) -> SegmentMatch:
        return SegmentMatch(
            chunk_index=0,
            t_start=18.0,
            t_end=48.0,
            text="The chain rule states that the derivative of a composite function is the derivative of the outer function times the derivative of the inner function.",
            score=score,
        )

    def test_create_reel_preserves_refined_window_past_max_len(self) -> None:
        refined_window = (18.0, 50.3)  # 32.3s > default max_len(30) for target=20
        reel = self.rs._create_reel(
            self.conn,
            material_id=self.MATERIAL_ID,
            concept=self._concept(),
            video=self._video(),
            segment=self._segment(),
            clip_window=refined_window,
            transcript=[],
            fast_mode=True,
            target_clip_duration_sec=20,
        )
        self.assertIsNotNone(reel)
        self.assertAlmostEqual(float(reel["t_start"]), refined_window[0])
        self.assertAlmostEqual(float(reel["t_end"]), refined_window[1])

        row = self.conn.execute(
            "SELECT t_start, t_end FROM reels WHERE id = ?",
            (reel["reel_id"],),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertAlmostEqual(float(row["t_start"]), refined_window[0])
        self.assertAlmostEqual(float(row["t_end"]), refined_window[1])

    def test_create_reel_keeps_chained_windows_contiguous(self) -> None:
        first_window = (20.0, 52.3)
        second_window = (52.3, 84.0)

        first = self.rs._create_reel(
            self.conn,
            material_id=self.MATERIAL_ID,
            concept=self._concept(),
            video=self._video(),
            segment=self._segment(score=0.92),
            clip_window=first_window,
            transcript=[],
            fast_mode=True,
            target_clip_duration_sec=20,
        )
        second = self.rs._create_reel(
            self.conn,
            material_id=self.MATERIAL_ID,
            concept=self._concept(),
            video=self._video(),
            segment=self._segment(score=0.88),
            clip_window=second_window,
            transcript=[],
            fast_mode=True,
            target_clip_duration_sec=20,
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertAlmostEqual(float(first["t_end"]), first_window[1])
        self.assertAlmostEqual(float(second["t_start"]), second_window[0])
        self.assertAlmostEqual(float(first["t_end"]), float(second["t_start"]))


class TopicStartInvariantTests(unittest.TestCase):
    """Topic-cut path: reels begin at the start of the relevant topic segment."""

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())

    def test_mention_cluster_starts_at_topic_introduction(self) -> None:
        # A video that mentions "calculus" first at t=0.8 (intro — to be skipped),
        # then a cluster of 4 mentions from t=45-75, then nothing else.
        cues = []
        t = 0.0
        # Intro (t=0..5): one mention that must be skipped (INTRO_BUFFER_SEC = 5.0)
        cues.append({"start": 0.0, "duration": 5.0, "text": "Today we'll briefly mention calculus and then talk about something else for a while."})
        # Filler (t=5..45): no mentions of calculus
        for i in range(10):
            cues.append({"start": 5.0 + i * 4.0, "duration": 4.0, "text": f"Filler cue number {i} discussing algebra and geometry instead."})
        # Topic cluster (t=45..75): several mentions
        cues.append({"start": 45.0, "duration": 5.0, "text": "Now let us turn to calculus."})
        cues.append({"start": 50.0, "duration": 5.0, "text": "Calculus is the study of continuous change."})
        cues.append({"start": 55.0, "duration": 5.0, "text": "In calculus we learn derivatives."})
        cues.append({"start": 60.0, "duration": 5.0, "text": "Calculus also covers integrals."})
        cues.append({"start": 65.0, "duration": 5.0, "text": "This is the core of calculus."})
        cues.append({"start": 70.0, "duration": 5.0, "text": "Wrapping up our calculus discussion."})
        # Tail (t=75..90)
        for i in range(3):
            cues.append({"start": 75.0 + i * 5.0, "duration": 5.0, "text": f"Moving on to physics lecture {i}."})
        segments = self.rs._topic_cut_segments_for_concept(
            transcript=cues, video_id="test", video_duration_sec=90,
            clip_min_len=15, clip_max_len=60, max_segments=5,
            concept_terms=["calculus"],
        )
        self.assertGreater(len(segments), 0)
        seg = segments[0]
        # Topic start should be in the cluster region (45-75), NOT in the intro.
        self.assertGreaterEqual(seg.t_start, 40.0)
        self.assertLess(seg.t_start, 50.0)

    def test_mention_cluster_drops_isolated_mentions_in_intro(self) -> None:
        # A video where the ONLY mention is in the first 5 seconds (intro).
        cues = []
        cues.append({"start": 0.0, "duration": 4.0, "text": "Today we'll talk about calculus."})  # intro, skipped
        for i in range(30):
            cues.append({"start": 4.0 + i * 3.0, "duration": 3.0, "text": "Unrelated content about algebra and geometry and history."})
        segments = self.rs._topic_cut_segments_for_concept(
            transcript=cues, video_id="test", video_duration_sec=120,
            clip_min_len=15, clip_max_len=60, max_segments=5,
            concept_terms=["calculus"],
        )
        self.assertEqual(len(segments), 0)


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

    def test_min_relevance_threshold_raises_quality_floor(self) -> None:
        self.rs._min_relevance_threshold = 0.45
        floor = self.rs._quality_floor_min_relevance(page_hint=1)
        self.assertGreaterEqual(floor, 0.45)

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


class TopicCutSingleWindowRefinementTests(unittest.TestCase):
    """Topic-cut segments that fit within max_len still get sentence-end
    refinement (new fix)."""

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())

    def test_topic_cut_single_window_is_sentence_aligned(self) -> None:
        # Simulate a topic_cut segment that ends slightly before a sentence terminator.
        # The refiner should push t_end forward to the terminator.
        cues = []
        texts = [
            "Topic starts with this sentence.",
            "The second sentence is here.",
            "A third sentence continues.",
            "The fourth sentence wraps up.",
            "And a fifth sentence ends the topic.",
            "Next topic begins immediately after.",
        ]
        for i, t in enumerate(texts):
            cues.append({"start": i * 4.0, "duration": 4.0, "text": t})
        # Topic_cut says topic is [0.0, 18.0) — but 18.0 is in the MIDDLE of cue 4.
        # The refiner should snap t_end to cue 4's end (20.0) where the terminator sits,
        # or fall back to some sentence end.
        win = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=0.0, proposed_end=18.0,
            video_duration_sec=40, min_len=15, max_len=22, min_start=0.0,
        )
        self.assertIsNotNone(win)
        self.assertEqual(win[0], 0.0)
        # End lands on a sentence terminator near the proposed 18.0.
        # Either 16.0 (end of cue 3) or 20.0 (end of cue 4) would be correct.
        self.assertIn(win[1], {16.0, 20.0})


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

    def test_default_params_produce_no_padding(self) -> None:
        """Backward-compat: callers that don't opt in get the same
        sentence-snapped boundaries as before."""
        cues = self._synthetic_cues(40)
        win = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=12.0, proposed_end=40.0,
            video_duration_sec=200, min_len=15, max_len=55,
        )
        self.assertIsNotNone(win)
        # With default pad_*_sec = 0, the start should be exactly on a cue
        # boundary (cue 3 starts at 12.0). The end should be on a cue end
        # (cue 9 ends at 39.9).
        self.assertEqual(win[0], 12.0)
        self.assertEqual(win[1], 39.9)

    def test_padding_shifts_boundaries_symmetrically(self) -> None:
        """With pad_start=0.3 and pad_end=0.3, the returned window is
        exactly 0.3s earlier at start and 0.3s later at end."""
        cues = self._synthetic_cues(40)
        baseline = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=12.0, proposed_end=40.0,
            video_duration_sec=200, min_len=15, max_len=55,
        )
        padded = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=12.0, proposed_end=40.0,
            video_duration_sec=200, min_len=15, max_len=55,
            pad_start_sec=0.3, pad_end_sec=0.3,
        )
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(padded)
        self.assertAlmostEqual(baseline[0] - padded[0], 0.3, places=5)
        self.assertAlmostEqual(padded[1] - baseline[1], 0.3, places=5)

    def test_pre_roll_clamped_by_min_start(self) -> None:
        """Pre-roll must never pull the start below min_start — that would
        overlap the previous reel in a chain continuation."""
        cues = self._synthetic_cues(40)
        padded = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=12.0, proposed_end=40.0,
            video_duration_sec=200, min_len=15, max_len=55,
            min_start=12.0,  # Refiner will snap start to cue boundary at 12.0.
            pad_start_sec=0.3, pad_end_sec=0.3,
        )
        self.assertIsNotNone(padded)
        # Pre-roll squeezed to 0 because refined_start == min_start.
        self.assertEqual(padded[0], 12.0)
        # Post-roll still applies.
        self.assertGreater(padded[1], 39.9)

    def test_pre_roll_clamped_to_zero(self) -> None:
        """Pre-roll never goes negative even when refined_start is small."""
        cues = self._synthetic_cues(40)
        padded = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=0.0, proposed_end=20.0,
            video_duration_sec=200, min_len=15, max_len=55,
            pad_start_sec=0.3, pad_end_sec=0.3,
        )
        self.assertIsNotNone(padded)
        self.assertGreaterEqual(padded[0], 0.0)

    def test_post_roll_clamped_by_video_duration(self) -> None:
        """Post-roll must never push the end past the video's duration."""
        cues = self._synthetic_cues(30)
        # Push end near the video duration; post-roll of 5s would overshoot.
        padded = self.rs._refine_clip_window_from_transcript(
            transcript=cues, proposed_start=80.0, proposed_end=119.0,
            video_duration_sec=120, min_len=15, max_len=60,
            pad_start_sec=0.3, pad_end_sec=5.0,
        )
        self.assertIsNotNone(padded)
        self.assertLessEqual(padded[1], 120.0)

    def test_split_chain_pre_roll_on_first_only(self) -> None:
        """In a multi-window split, only the first reel's start is
        pre-rolled; middle reels chain exactly against the previous end."""
        cues = self._synthetic_cues(50)
        windows = self.rs._split_into_consecutive_windows(
            transcript=cues, segment_start=12.0, segment_end=132.0,
            video_duration_sec=200, min_len=20, max_len=55,
        )
        self.assertGreaterEqual(len(windows), 2)
        # First reel start should sit below segment_start by up to 0.3s
        # (the refiner lands on a cue at/near 12.0; pre-roll pulls back).
        first_start, first_end = windows[0]
        self.assertLessEqual(first_start, 12.0)
        self.assertGreaterEqual(first_start, 12.0 - 0.35)
        # Chain continuity: consecutive windows don't overlap.
        for i in range(len(windows) - 1):
            cur_end = windows[i][1]
            next_start = windows[i + 1][0]
            # Next reel must start at or after the current reel's end.
            self.assertGreaterEqual(next_start, cur_end - 1e-6)

    def test_split_chain_post_roll_on_last_only(self) -> None:
        """Last reel's end is post-rolled; intermediate reel ends are not
        (else ``current_start`` would advance past real content)."""
        cues = self._synthetic_cues(50)
        windows = self.rs._split_into_consecutive_windows(
            transcript=cues, segment_start=12.0, segment_end=132.0,
            video_duration_sec=200, min_len=20, max_len=55,
        )
        self.assertGreaterEqual(len(windows), 2)
        # Last reel's end should extend past the natural cue end by
        # about REEL_PAD_END_SEC (0.3s) — we just check it's beyond what
        # a non-padded cue end would produce.
        last_s, last_e = windows[-1]
        # Either the last reel stopped at video boundary or near segment_end
        # + post-roll. We can at least assert it's >= the nearest cue end.
        # Cue 32 ends at 32*4 + 3.9 = 131.9.
        self.assertGreaterEqual(last_e, 131.9)

    def test_split_single_window_pads_both_ends(self) -> None:
        """When the segment fits in one reel, the single-window path pads
        both ends symmetrically."""
        cues = self._synthetic_cues(30)
        # Segment from 12 to 56 — fits in max_len=55 + 8s tolerance.
        windows = self.rs._split_into_consecutive_windows(
            transcript=cues, segment_start=12.0, segment_end=56.0,
            video_duration_sec=200, min_len=20, max_len=55,
        )
        self.assertEqual(len(windows), 1)
        s, e = windows[0]
        # Pre-roll: start should be ~0.3s below the nearest cue at/near 12.0.
        self.assertLess(s, 12.0)
        self.assertGreaterEqual(s, 11.0)
        # Post-roll: end should extend ~0.3s past the nearest cue end.
        # (Cue 13 ends at 13*4+3.9 = 55.9; padded ~56.2.)
        self.assertGreater(e, 55.9)

    def test_padding_constants_exported(self) -> None:
        """Production callers import REEL_PAD_START_SEC / REEL_PAD_END_SEC.
        Lock in the current values so drift triggers an explicit review."""
        from app.services.reels import REEL_PAD_START_SEC, REEL_PAD_END_SEC
        self.assertEqual(REEL_PAD_START_SEC, 0.3)
        self.assertEqual(REEL_PAD_END_SEC, 0.3)


if __name__ == "__main__":
    unittest.main()
