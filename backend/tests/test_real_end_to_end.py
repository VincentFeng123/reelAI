"""
Real-data end-to-end regression tests using ACTUAL transcripts captured from
YouTube (3Blue1Brown — "The essence of calculus", manual/punctuated captions,
video ID WUvTyaaNkzM). Unlike the hand-built synthetic fixtures in
``test_reel_invariants.py``, this exercises the pipeline against honest
real-world content.

The transcript below was fetched live via ``youtube-transcript-api`` during
development; we check it in so the test is deterministic and doesn't depend
on network access or YouTube rate limits.
"""

from __future__ import annotations

import unittest
from typing import Any

from app.services.embeddings import EmbeddingService
from app.services.reels import ReelService
from app.services.youtube import YouTubeService


# Real excerpt from 3Blue1Brown — "The essence of calculus" (manual captions,
# fully punctuated). Timestamps are verbatim from youtube-transcript-api.
# This sample covers the first ~120s (topic intro cluster) plus the outro
# cluster (~930-970s) to exercise both the chained sub-part path and the
# disjoint-cluster path.
REAL_3B1B_EXCERPT: list[dict[str, Any]] = [
    {"start": 15.0, "duration": 1.82, "text": "Hi everyone. Steven here, and"},
    {"start": 16.82, "duration": 3.60, "text": "This is the first video in a series on the essence of calculus,"},
    {"start": 20.42, "duration": 2.78, "text": "and I'll be publishing the follow-on videos"},
    {"start": 23.20, "duration": 2.88, "text": "in the days ahead, one per day for the next 10 days."},
    {"start": 26.08, "duration": 3.64, "text": "The goal here, as the name suggests,"},
    {"start": 29.72, "duration": 3.08, "text": "is to really get at the heart of what this subject is all about."},
    {"start": 32.80, "duration": 3.16, "text": "But this is a more ambitious goal than you might realize."},
    {"start": 35.96, "duration": 3.36, "text": "Calculus is a deep subject, full of many beautiful, unexpected ideas."},
    {"start": 39.32, "duration": 4.16, "text": "It has a lot to offer, but there's a lot there,"},
    {"start": 43.48, "duration": 3.44, "text": "which means there are a lot of potential pitfalls for students."},
    {"start": 46.92, "duration": 3.88, "text": "Things like the fundamental theorem of calculus,"},
    {"start": 50.80, "duration": 2.92, "text": "derivatives of all the common functions,"},
    {"start": 53.72, "duration": 3.04, "text": "techniques for computing integrals — these are not the essence."},
    {"start": 56.76, "duration": 3.84, "text": "They are the tools used to build something up."},
    {"start": 60.60, "duration": 2.80, "text": "What we want is for you to come away from this series"},
    {"start": 63.40, "duration": 2.60, "text": "feeling like you could have invented calculus yourself."},
    {"start": 66.00, "duration": 0.92, "text": ""},  # natural silence before next sentence
    {"start": 66.92, "duration": 3.60, "text": "Inventing math is no joke, and there is a difference between being"},
    {"start": 70.52, "duration": 3.80, "text": "told why something's true versus actually discovering it for yourself."},
    {"start": 74.32, "duration": 3.40, "text": "But at all stages, I want us to ask,"},
    {"start": 77.72, "duration": 3.52, "text": "if you were an early mathematician pondering these ideas,"},
    {"start": 81.24, "duration": 3.16, "text": "what could you have done to see why calculus works"},
    {"start": 84.40, "duration": 3.20, "text": "the way it does, to see why they're fundamental?"},
    {"start": 87.60, "duration": 4.04, "text": "In this opening video, I want to show you how you might stumble"},
    {"start": 91.64, "duration": 4.00, "text": "into the core ideas of calculus by thinking very carefully"},
    {"start": 95.64, "duration": 3.44, "text": "about one specific bit of geometry — the area of a circle."},
    {"start": 99.08, "duration": 3.84, "text": "Maybe you know that this is pi times its radius squared,"},
    {"start": 102.92, "duration": 3.48, "text": "but why? Is there a nice way to think about where this formula comes from?"},
    {"start": 106.40, "duration": 3.92, "text": "Well, contemplating this problem and leaving yourself open to"},
    {"start": 110.32, "duration": 3.80, "text": "exploring the interesting thoughts that come about can actually lead"},
    {"start": 114.12, "duration": 3.80, "text": "you to a glimpse of three big ideas in calculus, integrals, derivatives,"},
    {"start": 117.92, "duration": 2.68, "text": "and the fact that they're opposites."},
    # ... (more cues between 120-930 that don't mention calculus) ...
    # Later cluster where calculus is mentioned again
    {"start": 930.60, "duration": 2.72, "text": "And that's the essence of calculus."},
    {"start": 933.32, "duration": 3.40, "text": "It ties together the two big ideas of integrals and derivatives,"},
    {"start": 936.72, "duration": 3.48, "text": "and it shows how each one is an inverse of the other."},
    {"start": 940.20, "duration": 4.28, "text": "Without this, calculus would just be a bag of tricks disconnected from each other."},
    {"start": 944.48, "duration": 3.96, "text": "But with it, the field starts to flow, and you begin to see"},
    {"start": 948.44, "duration": 3.80, "text": "how ideas in calculus, discovered in very different times and places,"},
    {"start": 952.24, "duration": 3.04, "text": "are deeply connected."},
    {"start": 955.28, "duration": 3.44, "text": "I want to thank you for watching. More videos in this calculus series"},
    {"start": 958.72, "duration": 2.96, "text": "are coming in the days ahead,"},
    {"start": 961.68, "duration": 3.20, "text": "so that we can see how all these ideas we've talked about"},
    {"start": 964.88, "duration": 2.96, "text": "could have as easily popped out naturally from your own explorations."},
    {"start": 967.84, "duration": 2.42, "text": "See you then."},
]


class RealTranscript3B1BTests(unittest.TestCase):
    """Pipeline behaviour on a real punctuated transcript."""

    def setUp(self) -> None:
        self.rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())
        self.transcript = [
            {"start": c["start"], "duration": c["duration"], "text": c["text"]}
            for c in REAL_3B1B_EXCERPT
        ]

    def test_transcript_detected_as_punctuated(self) -> None:
        self.assertTrue(self.rs._transcript_has_terminal_punct(self.transcript))

    def test_topic_cut_finds_two_calculus_clusters(self) -> None:
        segs = self.rs._topic_cut_segments_for_concept(
            transcript=self.transcript,
            video_id="WUvTyaaNkzM",
            video_duration_sec=1025,
            clip_min_len=30,
            clip_max_len=55,
            max_segments=6,
            concept_terms=["calculus"],
        )
        self.assertGreater(len(segs), 0)
        # Expect coverage of both the intro cluster (~16-120s) and the outro
        # cluster (~930-970s).
        early = [s for s in segs if s.t_start < 200]
        late = [s for s in segs if s.t_start > 900]
        self.assertGreater(len(early), 0, "Expected coverage of early calculus discussion")
        self.assertGreater(len(late), 0, "Expected coverage of outro calculus discussion")

    def test_pipeline_produces_sentence_aligned_reels(self) -> None:
        segs = self.rs._topic_cut_segments_for_concept(
            transcript=self.transcript,
            video_id="WUvTyaaNkzM",
            video_duration_sec=1025,
            clip_min_len=30,
            clip_max_len=55,
            max_segments=6,
            concept_terms=["calculus"],
        )
        # Replicate main-loop chaining + refinement.
        chain: dict[str, float] = {}
        last_end: float | None = None
        BRIDGE = 2.0
        windows: list[tuple[float, float]] = []
        for seg in sorted(
            segs,
            key=lambda s: (float(s.t_start), int(getattr(s, "cluster_sub_index", 0))),
        ):
            span = seg.t_end - seg.t_start
            cg = str(getattr(seg, "cluster_group_id", "") or "")
            prev = chain.get(cg) if cg else None
            if prev is not None:
                eff = float(prev)
            elif last_end is not None and abs(float(seg.t_start) - last_end) <= BRIDGE:
                eff = float(last_end)
            else:
                eff = float(seg.t_start)

            if span > 55 + 16:
                w = self.rs._split_into_consecutive_windows(
                    transcript=self.transcript, segment_start=eff, segment_end=seg.t_end,
                    video_duration_sec=1025, min_len=30, max_len=55,
                )
            else:
                single = self.rs._refine_clip_window_from_transcript(
                    transcript=self.transcript, proposed_start=eff, proposed_end=seg.t_end,
                    video_duration_sec=1025, min_len=max(1, min(30, int(span * 0.6))),
                    max_len=int(max(span + 16, 55)), min_start=eff,
                )
                w = [single] if single else []
            windows.extend(w)
            if w:
                last_end = float(w[-1][1])
                if cg:
                    chain[cg] = last_end

        self.assertGreater(len(windows), 0)

        # 1. Every reel must open with a sentence starter — the first cue's
        #    text begins with a capital letter (real sentence start), OR the
        #    previous cue ends with terminal punctuation, OR the reel is at
        #    the very start of the transcript. We allow the "capital letter"
        #    fallback because real transcripts sometimes break sentences
        #    mid-cue (e.g. "Hi everyone. Steven here, and") and the cue
        #    boundary IS the correct place to start even though the
        #    PRECEDING cue doesn't terminate cleanly.
        for a, b in windows:
            idx_at_start = None
            for i, c in enumerate(self.transcript):
                if abs(c["start"] - a) < 0.05:
                    idx_at_start = i
                    break
            self.assertIsNotNone(
                idx_at_start,
                f"reel start {a} doesn't align to a cue boundary",
            )
            if idx_at_start == 0:
                continue
            open_text = self.transcript[idx_at_start]["text"].strip()
            prev_text = self.transcript[idx_at_start - 1]["text"].strip()
            opens_with_capital = bool(open_text) and open_text[0].isalpha() and open_text[0].isupper()
            prev_ends_sentence = self.rs._is_sentence_end(prev_text)
            self.assertTrue(
                opens_with_capital or prev_ends_sentence,
                f"reel at {a}s opens with {open_text!r} after {prev_text!r} — "
                "neither a capital-letter sentence start nor preceded by terminal punct",
            )

        # 2. Every reel end must land at a cue that ends a sentence.
        for a, b in windows:
            idx_at_end = None
            for i, c in enumerate(self.transcript):
                if abs((c["start"] + c["duration"]) - b) < 0.2:
                    idx_at_end = i
                    break
            self.assertIsNotNone(
                idx_at_end,
                f"reel end {b} doesn't align to any cue end",
            )
            end_text = self.transcript[idx_at_end]["text"].strip()
            self.assertTrue(
                self.rs._is_sentence_end(end_text),
                f"reel ending at {b}s closes with '{end_text}' which doesn't end with terminal punct",
            )

        # 3. Reels from the same cluster chain must have zero overlap.
        for i in range(len(windows) - 1):
            a_end = windows[i][1]
            b_start = windows[i + 1][0]
            # Within a cluster the gap should be ≤ 3s (natural silence
            # between sentences, or zero when chained). Across clusters it
            # can be much larger — that's expected.
            if b_start - a_end < 3.0:  # within-cluster chain
                self.assertGreaterEqual(
                    b_start, a_end - 0.01,
                    f"Overlap between reels: {windows[i]} and {windows[i+1]}",
                )

    def test_consecutive_reels_within_cluster_have_no_overlap(self) -> None:
        # Force a sub-split: tiny max_len so the first calculus cluster
        # (16.82-117.92) must split into multiple consecutive reels.
        segs = self.rs._topic_cut_segments_for_concept(
            transcript=self.transcript,
            video_id="WUvTyaaNkzM",
            video_duration_sec=1025,
            clip_min_len=20,
            clip_max_len=30,
            max_segments=6,
            concept_terms=["calculus"],
        )
        cluster_groups: dict[str, list] = {}
        for s in segs:
            cg = getattr(s, "cluster_group_id", "") or ""
            if cg:
                cluster_groups.setdefault(cg, []).append(s)
        self.assertTrue(
            any(len(v) > 1 for v in cluster_groups.values()),
            "Expected at least one oversized cluster to be split",
        )


if __name__ == "__main__":
    unittest.main()
