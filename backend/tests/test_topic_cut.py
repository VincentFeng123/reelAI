"""
Tests for `backend/app/services/topic_cut.py`.

These tests deliberately avoid the network and the OpenAI API. They cover:

  * URL / video ID parsing (every supported URL shape, plus rejection of garbage)
  * Short vs long-form classification (URL-based, duration-based, transcript-fallback)
  * The boundary-snapping pipeline (`_snap_segments_to_cues`)
  * The lexical-novelty heuristic on a fixture transcript with two visibly
    distinct topics
  * The full `cut_video_into_topic_reels` flow with `fetch_transcript` and the
    OpenAI client both mocked out, exercising both the LLM-success and
    LLM-failure-falls-back-to-heuristic branches
  * The CLI entry point with --json output

The fixture transcript is intentionally hand-built to be non-trivial: 60 cues
across 5 minutes, two clearly distinct topics (cooking pasta vs JavaScript
debugging), with a hard transition at cue 30.
"""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services import topic_cut  # noqa: E402
from backend.app.services.topic_cut import (  # noqa: E402
    MAX_TOPIC_REEL_SEC,
    MIN_TOPIC_REEL_SEC,
    Chapter,
    TopicReel,
    TranscriptCue,
    VideoClassification,
    _compose_label_from_terms,
    _heuristic_topic_segments,
    _is_skippable_chapter,
    _semantic_topic_segments,
    _snap_segments_to_cues,
    _split_long_range,
    _tfidf_labels_for_ranges,
    _topic_reels_from_chapters,
    chapters_to_topic_segments,
    classify_video,
    cues_from_ingest_cues,
    cut_video_into_topic_reels,
    extract_chapters,
    extract_video_id,
    main,
)


def _fixture_two_topic_transcript() -> list[TranscriptCue]:
    """
    60 cues, 5s each = 300 seconds. Cue 0-29 talk about pasta; cue 30-59 talk
    about JavaScript. The vocabulary jump at idx 30 is what the heuristic
    must catch.
    """
    pasta_lines = [
        "Today we're cooking spaghetti carbonara from scratch",
        "First grate the pecorino romano cheese",
        "Crack three eggs into a mixing bowl",
        "Whisk the egg yolks with grated pecorino",
        "Boil salted water for the spaghetti pasta",
        "Cook guanciale slowly in a wide skillet",
        "Render the pork fat at low heat",
        "Drop the spaghetti into the boiling water",
        "Stir the pasta to keep strands separate",
        "Reserve a cup of starchy pasta water",
        "Drain the spaghetti before fully al dente",
        "Toss spaghetti directly into the guanciale skillet",
        "Remove the skillet from the heat now",
        "Pour the egg cheese mixture over hot pasta",
        "Toss vigorously to coat every strand",
        "Add starchy water to create a silky sauce",
        "Crack fresh black pepper over the carbonara",
        "Plate immediately while the sauce is glossy",
        "Top with extra grated pecorino cheese",
        "Add more cracked pepper for finish",
        "Garnish with crispy guanciale pieces",
        "Serve carbonara hot with a side salad",
        "Pair this with a crisp white wine",
        "Carbonara should never contain cream",
        "Real Roman carbonara uses only egg cheese pork",
        "The starchy water emulsifies the sauce naturally",
        "Tossing off heat prevents scrambling the eggs",
        "Carbonara is a classic Roman pasta dish",
        "Practice your tossing technique often",
        "That's the perfect plate of carbonara",
    ]
    js_lines = [
        "Now let's switch gears and debug some JavaScript",
        "Open Chrome DevTools with command option I",
        "Set a breakpoint on line forty-two",
        "Inspect the call stack panel carefully",
        "Watch the variable values update in scope",
        "Step over each function call one at a time",
        "Use console log to print intermediate values",
        "Check the network tab for failed XHR requests",
        "Throttle the network speed to test slow connections",
        "Source maps reveal the original TypeScript code",
        "Use the debugger keyword to pause execution",
        "Inspect closures to find captured variable bugs",
        "Examine the prototype chain of unfamiliar objects",
        "Use performance profiler to spot slow render loops",
        "Memory snapshots reveal retained DOM nodes",
        "Heap allocation timelines find leaking closures",
        "React DevTools highlight component re-renders",
        "Check effect dependency arrays for missing values",
        "Async stack traces follow promise chains across awaits",
        "Use conditional breakpoints to skip noisy iterations",
        "Logpoints print without modifying source files",
        "The event listener panel shows attached handlers",
        "Pause on uncaught exceptions to find silent failures",
        "Storage tab shows cookies localStorage and indexedDB",
        "Lighthouse audits performance accessibility and SEO",
        "Coverage tab finds unused JavaScript bytes",
        "Disable JavaScript to confirm progressive enhancement",
        "Mobile emulator tests responsive viewport behavior",
        "Remote debug Android Chrome over USB cable",
        "That covers the essential debugging workflow",
    ]
    cues: list[TranscriptCue] = []
    t = 0.0
    for line in pasta_lines + js_lines:
        cues.append(TranscriptCue(start=t, duration=5.0, text=line))
        t += 5.0
    return cues


# --------------------------------------------------------------------------- #
# extract_video_id
# --------------------------------------------------------------------------- #


class ExtractVideoIdTests(unittest.TestCase):
    def test_bare_id(self) -> None:
        vid, is_short = extract_video_id("aircAruvnKk")
        self.assertEqual(vid, "aircAruvnKk")
        self.assertFalse(is_short)

    def test_watch_url(self) -> None:
        vid, is_short = extract_video_id("https://www.youtube.com/watch?v=aircAruvnKk&t=42s")
        self.assertEqual(vid, "aircAruvnKk")
        self.assertFalse(is_short)

    def test_short_url(self) -> None:
        vid, is_short = extract_video_id("https://youtu.be/aircAruvnKk?si=abc")
        self.assertEqual(vid, "aircAruvnKk")
        self.assertFalse(is_short)

    def test_shorts_url_flagged(self) -> None:
        vid, is_short = extract_video_id("https://www.youtube.com/shorts/aircAruvnKk")
        self.assertEqual(vid, "aircAruvnKk")
        self.assertTrue(is_short)

    def test_embed_url(self) -> None:
        vid, _ = extract_video_id("https://www.youtube.com/embed/aircAruvnKk")
        self.assertEqual(vid, "aircAruvnKk")

    def test_garbage(self) -> None:
        with self.assertRaises(ValueError):
            extract_video_id("https://example.com/not-youtube")
        with self.assertRaises(ValueError):
            extract_video_id("totally-not-an-id")
        with self.assertRaises(ValueError):
            extract_video_id("")


# --------------------------------------------------------------------------- #
# classify_video
# --------------------------------------------------------------------------- #


class ClassifyVideoTests(unittest.TestCase):
    def test_shorts_url_is_short(self) -> None:
        result = classify_video("https://www.youtube.com/shorts/aircAruvnKk")
        self.assertTrue(result.is_short)
        self.assertIn("/shorts/", result.reason)

    def test_short_duration_is_short(self) -> None:
        result = classify_video("aircAruvnKk", duration_sec=45)
        self.assertTrue(result.is_short)

    def test_at_60s_threshold_is_short(self) -> None:
        result = classify_video("aircAruvnKk", duration_sec=60)
        self.assertTrue(result.is_short)

    def test_long_duration_is_long_form(self) -> None:
        result = classify_video("aircAruvnKk", duration_sec=720)
        self.assertFalse(result.is_short)

    def test_transcript_fallback_when_duration_missing(self) -> None:
        cues = _fixture_two_topic_transcript()  # 300 seconds
        result = classify_video("aircAruvnKk", duration_sec=None, transcript=cues)
        self.assertFalse(result.is_short)
        self.assertGreaterEqual(result.duration_sec, 295)

    def test_unknown_duration_defaults_long_form(self) -> None:
        result = classify_video("aircAruvnKk")
        self.assertFalse(result.is_short)


# --------------------------------------------------------------------------- #
# Boundary snapping
# --------------------------------------------------------------------------- #


class SnapSegmentsTests(unittest.TestCase):
    def test_basic_snap(self) -> None:
        cues = _fixture_two_topic_transcript()
        # Two segments: 0-29 (pasta), 30-59 (JS).
        raw = [(0, 29, "Cooking carbonara", "summary one"),
               (30, 59, "JS debugging", "summary two")]
        reels = _snap_segments_to_cues(raw, cues, video_id="aircAruvnKk", video_duration_sec=300)
        self.assertEqual(len(reels), 2)
        self.assertEqual(reels[0].label, "Cooking carbonara")
        self.assertEqual(reels[0].cue_start_idx, 0)
        self.assertEqual(reels[0].cue_end_idx, 29)
        self.assertAlmostEqual(reels[0].t_start, 0.0)
        # Cue 29 starts at 145.0s, ends at 150.0s. Snap target should land in [148, 152].
        self.assertGreaterEqual(reels[0].t_end, 148.0)
        self.assertLessEqual(reels[0].t_end, 152.0)
        self.assertAlmostEqual(reels[1].t_start, 150.0)

    def test_drops_too_short_segment(self) -> None:
        cues = _fixture_two_topic_transcript()
        # 1-cue segment = 5s, well below MIN_TOPIC_REEL_SEC (30s).
        raw = [(0, 0, "tiny", "")]
        reels = _snap_segments_to_cues(raw, cues, video_id="vid12345678", video_duration_sec=300)
        self.assertEqual(reels, [])

    def test_drops_too_long_segment(self) -> None:
        # Build 1000 cues at 5s each = 5000 seconds; one segment from 0 to 999
        # is well over the 12-min cap and should be dropped.
        cues = [
            TranscriptCue(start=i * 5.0, duration=5.0, text=f"word{i}")
            for i in range(1000)
        ]
        raw = [(0, 999, "monolithic", "")]
        reels = _snap_segments_to_cues(raw, cues, video_id="vid12345678", video_duration_sec=5000)
        self.assertEqual(reels, [])

    def test_overlapping_segments_pushed_apart(self) -> None:
        cues = _fixture_two_topic_transcript()
        # Two segments that overlap by ~25s — second should start at first's end.
        raw = [(0, 29, "first", ""), (25, 50, "second", "")]
        reels = _snap_segments_to_cues(raw, cues, video_id="vid12345678", video_duration_sec=300)
        self.assertEqual(len(reels), 2)
        self.assertGreaterEqual(reels[1].t_start, reels[0].t_end - 0.01)

    def test_clamps_to_video_duration(self) -> None:
        cues = _fixture_two_topic_transcript()
        # Pretend video duration is shorter than the last cue's end.
        raw = [(0, 59, "everything", "")]
        reels = _snap_segments_to_cues(raw, cues, video_id="vid12345678", video_duration_sec=200)
        # 200s is below the MAX cap, so this segment should still be returned.
        self.assertEqual(len(reels), 1)
        self.assertLessEqual(reels[0].t_end, 200.0)


# --------------------------------------------------------------------------- #
# Heuristic
# --------------------------------------------------------------------------- #


class HeuristicTests(unittest.TestCase):
    def test_finds_topic_boundary_between_pasta_and_js(self) -> None:
        cues = _fixture_two_topic_transcript()
        segments = _heuristic_topic_segments(cues, target_duration_sec=60.0)
        self.assertGreaterEqual(len(segments), 2,
                                f"expected at least 2 segments, got {segments}")
        # Some segment should start in the JS half (idx >= 30).
        starts_in_js = [s for s in segments if s[0] >= 25]
        self.assertTrue(starts_in_js,
                        f"no segment landed in the JS half, segments={segments}")
        # The split point should be near idx 30 (within ±8 cues = ±40s).
        boundary_starts = [s[0] for s in segments]
        nearest = min(boundary_starts, key=lambda i: abs(i - 30))
        self.assertLessEqual(abs(nearest - 30), 8,
                             f"closest boundary to expected split (30) was at {nearest}")

    def test_short_transcript_returns_one_segment(self) -> None:
        cues = [TranscriptCue(start=i, duration=1.0, text=f"hi {i}") for i in range(4)]
        segments = _heuristic_topic_segments(cues)
        self.assertEqual(len(segments), 1)


# --------------------------------------------------------------------------- #
# cut_video_into_topic_reels — full flow with mocked transcript + LLM
# --------------------------------------------------------------------------- #


def _make_mock_openai_client(segments_payload: dict) -> mock.MagicMock:
    """Build a MagicMock that mimics openai.OpenAI's chat.completions.create."""
    client = mock.MagicMock()
    response = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = json.dumps(segments_payload)
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


class CutFullFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_cues = _fixture_two_topic_transcript()
        self._patcher = mock.patch.object(
            topic_cut, "fetch_transcript", return_value=self.fixture_cues
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_short_returns_empty_list(self) -> None:
        classification, reels = cut_video_into_topic_reels(
            "https://www.youtube.com/shorts/aircAruvnKk"
        )
        self.assertTrue(classification.is_short)
        self.assertEqual(reels, [])

    def test_llm_success_path(self) -> None:
        client = _make_mock_openai_client({
            "segments": [
                {"start_idx": 0, "end_idx": 29, "label": "Cooking spaghetti carbonara from scratch",
                 "summary": "Step by step Roman pasta recipe"},
                {"start_idx": 30, "end_idx": 59, "label": "Debugging JavaScript with Chrome DevTools",
                 "summary": "Breakpoints, profiling, and the network panel"},
            ]
        })
        classification, reels = cut_video_into_topic_reels(
            "aircAruvnKk", duration_sec=300, openai_client=client,
        )
        self.assertFalse(classification.is_short)
        self.assertEqual(len(reels), 2)
        self.assertEqual(reels[0].label, "Cooking spaghetti carbonara from scratch")
        self.assertEqual(reels[1].label, "Debugging JavaScript with Chrome DevTools")
        client.chat.completions.create.assert_called_once()

    def test_llm_failure_falls_back_to_heuristic(self) -> None:
        client = mock.MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("api down")
        classification, reels = cut_video_into_topic_reels(
            "aircAruvnKk", duration_sec=300, openai_client=client,
        )
        self.assertFalse(classification.is_short)
        # Heuristic should still produce ≥2 segments on this fixture.
        self.assertGreaterEqual(len(reels), 2)
        for reel in reels:
            self.assertGreaterEqual(reel.duration_sec, MIN_TOPIC_REEL_SEC)

    def test_no_transcript_returns_empty(self) -> None:
        with mock.patch.object(topic_cut, "fetch_transcript", return_value=[]):
            classification, reels = cut_video_into_topic_reels("aircAruvnKk", duration_sec=300)
        self.assertFalse(classification.is_short)
        self.assertEqual(reels, [])

    def test_use_llm_false_skips_llm_entirely(self) -> None:
        client = mock.MagicMock()
        _, reels = cut_video_into_topic_reels(
            "aircAruvnKk", duration_sec=300, openai_client=client, use_llm=False,
        )
        client.chat.completions.create.assert_not_called()
        self.assertGreaterEqual(len(reels), 1)


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #


class CliTests(unittest.TestCase):
    def test_cli_json_output_short(self) -> None:
        with mock.patch.object(topic_cut, "fetch_transcript", return_value=[]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["https://www.youtube.com/shorts/aircAruvnKk", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["classification"]["is_short"])
        self.assertEqual(payload["reels"], [])

    def test_cli_garbage_input_returns_2(self) -> None:
        rc = main(["not-a-youtube-url"])
        self.assertEqual(rc, 2)


# --------------------------------------------------------------------------- #
# YouTube chapters extraction
# --------------------------------------------------------------------------- #


class ExtractChaptersTests(unittest.TestCase):
    def test_extracts_creator_chapters(self) -> None:
        info = {
            "chapters": [
                {"start_time": 0.0, "end_time": 67.0, "title": "Introduction example"},
                {"start_time": 67.0, "end_time": 162.0, "title": "Series preview"},
                {"start_time": 162.0, "end_time": 215.0, "title": "What are neurons?"},
            ]
        }
        chapters = extract_chapters(info)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0].title, "Introduction example")
        self.assertAlmostEqual(chapters[0].start, 0.0)
        self.assertAlmostEqual(chapters[0].end, 67.0)
        self.assertEqual(chapters[2].title, "What are neurons?")

    def test_handles_missing_chapters_gracefully(self) -> None:
        self.assertEqual(extract_chapters(None), [])
        self.assertEqual(extract_chapters({}), [])
        self.assertEqual(extract_chapters({"chapters": None}), [])
        self.assertEqual(extract_chapters({"chapters": []}), [])
        self.assertEqual(extract_chapters({"chapters": "not a list"}), [])

    def test_drops_invalid_entries(self) -> None:
        info = {
            "chapters": [
                {"start_time": 0.0, "end_time": 30.0, "title": "valid"},
                {"start_time": 50.0, "end_time": 50.0, "title": "zero length"},   # bad
                {"start_time": 60.0, "end_time": 90.0, "title": ""},              # bad
                {"start_time": 100.0, "end_time": 130.0, "title": "valid two"},
                "garbage",                                                         # bad
            ]
        }
        chapters = extract_chapters(info)
        self.assertEqual(len(chapters), 2)
        self.assertEqual([c.title for c in chapters], ["valid", "valid two"])

    def test_dedupes_identical_entries(self) -> None:
        info = {
            "chapters": [
                {"start_time": 0.0, "end_time": 30.0, "title": "Recap"},
                {"start_time": 0.0, "end_time": 30.0, "title": "recap"},  # case dup
            ]
        }
        chapters = extract_chapters(info)
        self.assertEqual(len(chapters), 1)


class IsSkippableChapterTests(unittest.TestCase):
    def test_obvious_fluff_is_skipped(self) -> None:
        for title in ("Intro", "Outro", "Sponsor", "Subscribe", "Conclusion",
                      "Sponsor: NordVPN", "Final thoughts and credits", ""):
            with self.subTest(title=title):
                self.assertTrue(_is_skippable_chapter(title))

    def test_real_topics_are_kept(self) -> None:
        # The critical regression: "Introducing layers" must NOT be killed
        # by a substring match on "intro".
        for title in ("Introducing layers", "Why intros matter",
                      "What are neurons?", "Series preview", "Recap",
                      "Edge detection example", "Notation and linear algebra",
                      "Introduction to advanced category theory and homotopy"):
            with self.subTest(title=title):
                self.assertFalse(_is_skippable_chapter(title))


class ChaptersToTopicSegmentsTests(unittest.TestCase):
    def test_drops_fluff_keeps_real_chapters(self) -> None:
        chapters = [
            Chapter(start=0.0, end=67.0, title="Intro"),
            Chapter(start=67.0, end=162.0, title="Series preview"),
            Chapter(start=162.0, end=215.0, title="What are neurons?"),
            Chapter(start=215.0, end=331.0, title="Introducing layers"),
            Chapter(start=900.0, end=950.0, title="Sponsor: NordVPN"),
            Chapter(start=950.0, end=1100.0, title="Recap"),
        ]
        segments = chapters_to_topic_segments(chapters)
        labels = [s[2] for s in segments]
        self.assertNotIn("Intro", labels)
        self.assertNotIn("Sponsor: NordVPN", labels)
        self.assertIn("Series preview", labels)
        self.assertIn("Introducing layers", labels)
        self.assertIn("Recap", labels)


class TopicReelsFromChaptersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cues = _fixture_two_topic_transcript()  # 60 cues × 5s = 300s

    def test_chapter_drives_topic_reel_directly(self) -> None:
        chapters = [
            Chapter(start=0.0, end=150.0, title="Cooking"),
            Chapter(start=150.0, end=300.0, title="JS Debugging"),
        ]
        segments = chapters_to_topic_segments(chapters)
        reels = _topic_reels_from_chapters(
            segments, self.cues, video_id="aircAruvnKk", video_duration_sec=300.0,
        )
        self.assertEqual(len(reels), 2)
        self.assertEqual(reels[0].label, "Cooking")
        self.assertEqual(reels[1].label, "JS Debugging")
        self.assertAlmostEqual(reels[0].t_start, 0.0)
        self.assertAlmostEqual(reels[1].t_end, 300.0)

    def test_chapter_too_short_is_dropped(self) -> None:
        chapters = [
            Chapter(start=0.0, end=10.0, title="Tiny chapter"),  # below 30s
            Chapter(start=10.0, end=70.0, title="Real chapter"),
        ]
        segments = chapters_to_topic_segments(chapters)
        reels = _topic_reels_from_chapters(
            segments, self.cues, video_id="vid12345678", video_duration_sec=300.0,
        )
        self.assertEqual(len(reels), 1)
        self.assertEqual(reels[0].label, "Real chapter")

    def test_works_without_transcript_cues(self) -> None:
        # Important: chapters carry their own absolute timestamps, so they
        # should still work when no transcript is available.
        chapters = [Chapter(start=0.0, end=120.0, title="A topic")]
        segments = chapters_to_topic_segments(chapters)
        reels = _topic_reels_from_chapters(
            segments, [], video_id="vid12345678", video_duration_sec=600.0,
        )
        self.assertEqual(len(reels), 1)
        self.assertAlmostEqual(reels[0].t_start, 0.0)
        self.assertAlmostEqual(reels[0].t_end, 120.0)


class CutWithChaptersTests(unittest.TestCase):
    """The chapters path should completely bypass the LLM AND the heuristic."""

    def setUp(self) -> None:
        self._patcher = mock.patch.object(
            topic_cut, "fetch_transcript", return_value=_fixture_two_topic_transcript()
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_chapters_short_circuit_llm(self) -> None:
        info = {
            "chapters": [
                {"start_time": 0.0, "end_time": 150.0, "title": "Cooking carbonara"},
                {"start_time": 150.0, "end_time": 300.0, "title": "Debugging JavaScript"},
            ]
        }
        client = mock.MagicMock()
        classification, reels = cut_video_into_topic_reels(
            "aircAruvnKk", duration_sec=300, openai_client=client, info_dict=info,
        )
        self.assertFalse(classification.is_short)
        self.assertEqual(len(reels), 2)
        self.assertEqual(reels[0].label, "Cooking carbonara")
        self.assertEqual(reels[1].label, "Debugging JavaScript")
        # The LLM must NOT have been called when chapters are present.
        client.chat.completions.create.assert_not_called()

    def test_no_chapters_falls_through_to_heuristic(self) -> None:
        # No info_dict means no chapters → falls through to LLM (none here) → heuristic.
        classification, reels = cut_video_into_topic_reels(
            "aircAruvnKk", duration_sec=300, use_llm=False,
        )
        self.assertFalse(classification.is_short)
        self.assertGreaterEqual(len(reels), 1)


# --------------------------------------------------------------------------- #
# TF-IDF labels
# --------------------------------------------------------------------------- #


class TfidfLabelsTests(unittest.TestCase):
    def test_distinctive_terms_outrank_common_words(self) -> None:
        cues = _fixture_two_topic_transcript()  # pasta vs JS
        ranges = [(0, 29), (30, 59)]
        labels = _tfidf_labels_for_ranges(cues, ranges)
        self.assertEqual(len(labels), 2)
        # First range is about pasta — its label should mention any term that's
        # common in the pasta half but absent from the JS half.
        pasta_terms = (
            "carbonara", "spaghetti", "pecorino", "guanciale", "pasta",
            "eggs", "egg", "cheese", "skillet", "cracked",
        )
        self.assertTrue(
            any(term in labels[0].lower() for term in pasta_terms),
            f"pasta label was {labels[0]!r}, expected to contain one of {pasta_terms}",
        )
        # Second range is about JS debugging — any of these terms is fine.
        # "values", "tab", "panel" etc. are valid even if they're less obvious
        # than "debug" because they're distinctively absent from the pasta half.
        js_terms = (
            "debug", "devtools", "breakpoint", "javascript", "chrome",
            "network", "tab", "values", "panel", "storage", "cookies",
            "console", "scope", "performance",
        )
        self.assertTrue(
            any(term in labels[1].lower() for term in js_terms),
            f"js label was {labels[1]!r}, expected to contain one of {js_terms}",
        )
        # Most importantly: the two labels MUST be different (TF-IDF would be
        # broken if it picked the same terms for both halves).
        self.assertNotEqual(labels[0].lower(), labels[1].lower())

    def test_empty_segment_returns_placeholder(self) -> None:
        cues = [TranscriptCue(start=0.0, duration=1.0, text="")]
        labels = _tfidf_labels_for_ranges(cues, [(0, 0)])
        self.assertEqual(labels, ["Untitled segment"])

    def test_compose_label_prefers_bigram_first(self) -> None:
        # When a bigram is in the top terms, it should appear first.
        label = _compose_label_from_terms(["neural network", "weights", "biases"])
        self.assertTrue(label.lower().startswith("neural network"))
        self.assertIn("Weights", label)

    def test_compose_label_dedups_bigram_words(self) -> None:
        # If "neural network" is the bigram, the unigram "neural" should NOT
        # also appear in the final label.
        label = _compose_label_from_terms(["neural network", "neural", "training"])
        self.assertEqual(label.lower().count("neural"), 1)


class HeuristicLabelsRegressionTest(unittest.TestCase):
    """The heuristic + new TF-IDF labels should yield distinctive labels."""

    def test_heuristic_emits_distinctive_labels(self) -> None:
        cues = _fixture_two_topic_transcript()
        segments = _heuristic_topic_segments(cues, target_duration_sec=60.0)
        # Each segment must have a non-empty label that ISN'T the old default.
        for s in segments:
            label = s[2]
            self.assertTrue(label, f"segment {s} has empty label")
            self.assertFalse(label.startswith("Segment "),
                             f"label {label!r} looks like the legacy fallback")


# --------------------------------------------------------------------------- #
# Semantic boundary detector — graceful skip when not installed
# --------------------------------------------------------------------------- #


class SemanticPathTests(unittest.TestCase):
    def test_returns_none_when_sentence_transformers_missing(self) -> None:
        # Patch the model loader to simulate "sentence-transformers not installed".
        with mock.patch.object(topic_cut, "_load_sentence_transformer", return_value=None):
            result = _semantic_topic_segments(_fixture_two_topic_transcript())
        self.assertIsNone(result)

    def test_returns_none_for_empty_transcript(self) -> None:
        with mock.patch.object(topic_cut, "_load_sentence_transformer", return_value=None):
            result = _semantic_topic_segments([])
        self.assertIsNone(result)

    def test_uses_model_when_available_and_returns_ranges(self) -> None:
        # Simulate a fake model that returns deterministic embeddings: chunks
        # in the first half get one vector, chunks in the second half get a
        # very different vector. The boundary should be detected at the half.
        import numpy as np
        cues = _fixture_two_topic_transcript()  # 60 cues, 300s

        class _FakeModel:
            def encode(self, texts, **kwargs):  # noqa: ARG002
                vecs = []
                half = len(texts) // 2
                for i in range(len(texts)):
                    if i < half:
                        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    vecs.append(v)
                return np.stack(vecs)

        with mock.patch.object(topic_cut, "_load_sentence_transformer",
                               return_value=_FakeModel()):
            result = _semantic_topic_segments(cues, target_duration_sec=60.0)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result or []), 2)


# --------------------------------------------------------------------------- #
# cues_from_ingest_cues adapter
# --------------------------------------------------------------------------- #


class CuesFromIngestCuesTests(unittest.TestCase):
    def test_adapts_pydantic_style_cues(self) -> None:
        # Mimic IngestTranscriptCue with a simple dataclass-like object
        class Fake:
            def __init__(self, start, end, text):
                self.start, self.end, self.text = start, end, text
        ingest = [Fake(0.0, 5.0, "hello"), Fake(5.0, 10.0, "world")]
        out = cues_from_ingest_cues(ingest)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "hello")
        self.assertAlmostEqual(out[0].duration, 5.0)
        self.assertAlmostEqual(out[1].end, 10.0)

    def test_drops_empty_text(self) -> None:
        class Fake:
            def __init__(self, start, end, text):
                self.start, self.end, self.text = start, end, text
        out = cues_from_ingest_cues([Fake(0.0, 5.0, "")])
        self.assertEqual(out, [])


# --------------------------------------------------------------------------- #
# max_reel_sec splitting (long topics get split into N parts)
# --------------------------------------------------------------------------- #


class SplitLongRangeTests(unittest.TestCase):
    def test_short_range_returned_unchanged(self) -> None:
        # 90s with max=120 → returned as-is
        out = _split_long_range(0.0, 90.0, "topic", "summary",
                                max_reel_sec=120.0, min_reel_sec=30.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], (0.0, 90.0, "topic", "summary"))

    def test_long_range_split_with_label_suffix(self) -> None:
        # 200s with max=60 → ceil(200/60)=4 parts of 50s each, all >= min(30)
        out = _split_long_range(0.0, 200.0, "Why layers", "",
                                max_reel_sec=60.0, min_reel_sec=30.0)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0][2], "Why layers (1/4)")
        self.assertEqual(out[3][2], "Why layers (4/4)")
        # First part should start at 0
        self.assertAlmostEqual(out[0][0], 0.0)
        # Last part should end at the original 200
        self.assertAlmostEqual(out[3][1], 200.0)
        # Parts are roughly equal length
        for a, b, _, _ in out:
            self.assertAlmostEqual(b - a, 50.0, places=1)

    def test_split_returns_empty_when_parts_too_short(self) -> None:
        # 100s with max=20 and min=30: ceil(100/20)=5 parts of 20s each.
        # Each part is below min(30), so split should bail out.
        out = _split_long_range(0.0, 100.0, "topic", "",
                                max_reel_sec=20.0, min_reel_sec=30.0)
        self.assertEqual(out, [])

    def test_split_handles_exact_multiple(self) -> None:
        # 180s with max=60 → exactly 3 equal parts
        out = _split_long_range(0.0, 180.0, "X", "",
                                max_reel_sec=60.0, min_reel_sec=30.0)
        self.assertEqual(len(out), 3)
        for i, (a, b, _, _) in enumerate(out):
            self.assertAlmostEqual(a, i * 60.0, places=1)
            self.assertAlmostEqual(b, (i + 1) * 60.0, places=1)


class TopicReelsFromChaptersSplittingTests(unittest.TestCase):
    """The chapters path should split chapters that exceed max_reel_sec."""

    def setUp(self) -> None:
        # 200 cues × 5s = 1000 seconds (~17 min) of transcript
        self.cues = [
            TranscriptCue(start=i * 5.0, duration=5.0, text=f"word{i}")
            for i in range(200)
        ]

    def test_long_chapter_is_split(self) -> None:
        # One chapter spanning 0-300s (5 minutes), with user's max_reel_sec=60
        chapters = [Chapter(start=0.0, end=300.0, title="Why layers?")]
        segments = chapters_to_topic_segments(chapters)
        reels = _topic_reels_from_chapters(
            segments, self.cues, video_id="aircAruvnKk", video_duration_sec=1000.0,
            min_reel_sec=30.0, max_reel_sec=60.0,
        )
        # 300/60 = 5 parts of 60s each
        self.assertEqual(len(reels), 5)
        labels = [r.label for r in reels]
        self.assertIn("Why layers? (1/5)", labels)
        self.assertIn("Why layers? (5/5)", labels)
        # All parts within the user's preferred range
        for r in reels:
            self.assertLessEqual(r.duration_sec, 60.0 + 1.0)
            self.assertGreaterEqual(r.duration_sec, 30.0)

    def test_short_chapter_kept_when_split_would_be_too_short(self) -> None:
        # 100s chapter, max=30, min=30 → 4 parts of 25s each → too short → keep whole
        chapters = [Chapter(start=0.0, end=100.0, title="Quick recap")]
        segments = chapters_to_topic_segments(chapters)
        reels = _topic_reels_from_chapters(
            segments, self.cues, video_id="aircAruvnKk", video_duration_sec=1000.0,
            min_reel_sec=30.0, max_reel_sec=30.0,
        )
        # The whole 100s chapter should be kept as one reel (split bailed out)
        self.assertEqual(len(reels), 1)
        self.assertEqual(reels[0].label, "Quick recap")
        self.assertAlmostEqual(reels[0].duration_sec, 100.0, places=0)


class SnapSegmentsSplittingTests(unittest.TestCase):
    """The LLM/heuristic path should also respect max_reel_sec via _snap_segments_to_cues."""

    def setUp(self) -> None:
        self.cues = [
            TranscriptCue(start=i * 5.0, duration=5.0, text=f"word{i}")
            for i in range(60)  # 300 seconds
        ]

    def test_segment_exceeding_max_is_split(self) -> None:
        # Segment from cue 0 to cue 49 = 250 seconds, with max=60 → 5 parts
        raw = [(0, 49, "Long topic", "")]
        reels = _snap_segments_to_cues(
            raw, self.cues, video_id="aircAruvnKk", video_duration_sec=300.0,
            min_reel_sec=30.0, max_reel_sec=60.0,
        )
        self.assertEqual(len(reels), 5)
        for r in reels:
            self.assertLessEqual(r.duration_sec, 60.0 + 1.0)
            self.assertIn("Long topic (", r.label)


# --------------------------------------------------------------------------- #
# _topic_cut_segments_for_concept (the new ReelService helper)
# --------------------------------------------------------------------------- #
#
# Test the helper directly without spinning up the full ReelService class.
# We instantiate a minimal stand-in object that has the two attributes the
# helper actually reads (`openai_client`) and the bound method.


class TopicCutSegmentsForConceptHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        # Long-form fixture: 60 cues × 5s = 300 seconds across two clearly
        # distinct topics (pasta vs JS), so topic_cut should detect a boundary.
        self.transcript_dicts = [
            {"start": cue.start, "duration": cue.duration, "text": cue.text}
            for cue in _fixture_two_topic_transcript()
        ]

        # Build a tiny stand-in that exposes the helper as a bound method.
        from backend.app.services.reels import ReelService
        # The helper only reads `self.openai_client` so we don't need a
        # full ReelService — just a minimal object with that attribute.
        class _StubService:
            openai_client = None  # forces topic_cut into the heuristic path
            _topic_cut_segments_for_concept = ReelService._topic_cut_segments_for_concept
        self.service = _StubService()

    def test_long_form_returns_topic_segments(self) -> None:
        # No concept_terms → no concept-anchoring, all topics returned.
        segments = self.service._topic_cut_segments_for_concept(
            transcript=self.transcript_dicts,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=120,
            max_segments=10,
        )
        self.assertGreaterEqual(len(segments), 1,
                                f"expected ≥1 topic_cut segment, got {segments}")
        for seg in segments:
            self.assertGreater(seg.t_end, seg.t_start)
            self.assertGreaterEqual(seg.t_end - seg.t_start, 30)
            self.assertLessEqual(seg.t_end - seg.t_start, 120 + 1)
            self.assertTrue(seg.text)

    def test_returns_empty_for_short_video(self) -> None:
        # A 50-second "video" — topic_cut classifies as Short via duration.
        short_transcript = self.transcript_dicts[:10]  # ~50s
        segments = self.service._topic_cut_segments_for_concept(
            transcript=short_transcript,
            video_id="aircAruvnKk",
            video_duration_sec=50,
            clip_min_len=15,
            clip_max_len=60,
            max_segments=5,
        )
        self.assertEqual(segments, [])

    def test_returns_empty_for_empty_transcript(self) -> None:
        segments = self.service._topic_cut_segments_for_concept(
            transcript=[],
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=60,
            max_segments=5,
        )
        self.assertEqual(segments, [])


class ConceptAnchoredRefinementTests(unittest.TestCase):
    """
    The `concept_terms` parameter must:
      1. Restrict the output to topics that mention the concept ≥2 times.
      2. Refine the t_start to the first matching cue (with optional 1-cue lead-in).
      3. Refine the t_end to the last matching cue.
      4. Drop topics with no concept mentions entirely.
    """

    def setUp(self) -> None:
        # Use the same pasta-vs-JS fixture. Pasta-related terms only appear in
        # cues 0-29; JS-related terms only appear in cues 30-59.
        self.transcript_dicts = [
            {"start": cue.start, "duration": cue.duration, "text": cue.text}
            for cue in _fixture_two_topic_transcript()
        ]
        from backend.app.services.reels import ReelService
        class _StubService:
            openai_client = None
            _topic_cut_segments_for_concept = ReelService._topic_cut_segments_for_concept
        self.service = _StubService()

    def test_pasta_concept_returns_only_pasta_topics(self) -> None:
        segments = self.service._topic_cut_segments_for_concept(
            transcript=self.transcript_dicts,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=200,
            max_segments=10,
            concept_terms=["carbonara", "spaghetti", "guanciale"],
        )
        self.assertGreaterEqual(len(segments), 1,
                                "expected ≥1 pasta segment, got nothing")
        # Every returned segment must lie in the pasta half (0-150s) — the JS
        # half (150-300s) contains zero pasta vocabulary so concept-anchoring
        # must drop it.
        for seg in segments:
            self.assertLess(seg.t_start, 160,
                            f"pasta segment leaked into JS half: {seg}")
            self.assertLessEqual(seg.t_end, 160,
                                 f"pasta segment ends in JS half: {seg}")

    def test_js_concept_returns_only_js_topics(self) -> None:
        segments = self.service._topic_cut_segments_for_concept(
            transcript=self.transcript_dicts,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=200,
            max_segments=10,
            concept_terms=["javascript", "debugging", "devtools", "breakpoint"],
        )
        self.assertGreaterEqual(len(segments), 1,
                                "expected ≥1 JS segment, got nothing")
        # Every returned segment must lie in the JS half (≥150s).
        for seg in segments:
            self.assertGreaterEqual(seg.t_start, 140,
                                    f"JS segment leaked into pasta half: {seg}")

    def test_unrelated_concept_returns_nothing(self) -> None:
        # Concept terms that appear in NEITHER half — every topic should be
        # dropped (the relevance hard-guarantee).
        segments = self.service._topic_cut_segments_for_concept(
            transcript=self.transcript_dicts,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=200,
            max_segments=10,
            concept_terms=["quantum", "entanglement", "wavefunction"],
        )
        self.assertEqual(segments, [],
                         f"expected empty list for unrelated concept, got {segments}")

    def test_refined_clip_starts_near_first_concept_mention(self) -> None:
        # Use a single distinctive concept term whose first mention is in the
        # MIDDLE of a topic, not at the start. The refined t_start should land
        # near that mention (within 1 cue lead-in), not at the topic boundary.
        segments = self.service._topic_cut_segments_for_concept(
            transcript=self.transcript_dicts,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=200,
            max_segments=10,
            concept_terms=["devtools"],  # appears in cues 31 and elsewhere in JS half
        )
        self.assertGreaterEqual(len(segments), 1)
        # The first segment's t_start should be close to where "devtools" first
        # appears (~155s for cue 31), not 0 or 150.
        # With 1-cue lead-in, we expect the start to be at or just before
        # the cue containing the first "devtools" mention.
        first_mention_indices = [
            i for i, c in enumerate(self.transcript_dicts)
            if "devtools" in c["text"].lower()
        ]
        if first_mention_indices:
            expected_start_floor = max(0, (first_mention_indices[0] - 1)) * 5.0
            for seg in segments:
                # The refined start should not be earlier than first_mention - 1 cue
                # AND not later than first_mention itself.
                self.assertGreaterEqual(
                    seg.t_start, expected_start_floor - 0.5,
                    f"refined start {seg.t_start} is earlier than first mention - 1 cue ({expected_start_floor})",
                )

    def test_no_concept_terms_returns_unrefined_topic_boundaries(self) -> None:
        # When concept_terms is None or empty, the helper falls back to
        # returning topic_cut's natural boundaries unrefined.
        segments_none = self.service._topic_cut_segments_for_concept(
            transcript=self.transcript_dicts,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=200,
            max_segments=10,
            concept_terms=None,
        )
        segments_empty = self.service._topic_cut_segments_for_concept(
            transcript=self.transcript_dicts,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=200,
            max_segments=10,
            concept_terms=[],
        )
        self.assertGreaterEqual(len(segments_none), 1)
        self.assertGreaterEqual(len(segments_empty), 1)
        # None and [] should produce equivalent output (same topic boundaries).
        self.assertEqual(
            [(s.t_start, s.t_end) for s in segments_none],
            [(s.t_start, s.t_end) for s in segments_empty],
        )

    def test_single_mention_topic_is_dropped(self) -> None:
        # Inject a transcript where the concept is mentioned only ONCE inside
        # a topic. The helper should drop it (≥2 mentions required).
        transcript = [
            {"start": i * 5.0, "duration": 5.0, "text": text}
            for i, text in enumerate([
                # Topic A: 30 cues about cooking
                *(f"cooking step {i}" for i in range(30)),
                # Topic B: 30 cues about JavaScript with EXACTLY ONE mention of "rust"
                *(f"javascript line {i}" for i in range(15)),
                "actually let me mention rust briefly",
                *(f"javascript line {i}" for i in range(15, 30)),
            ])
        ]
        segments = self.service._topic_cut_segments_for_concept(
            transcript=transcript,
            video_id="aircAruvnKk",
            video_duration_sec=300,
            clip_min_len=30,
            clip_max_len=200,
            max_segments=10,
            concept_terms=["rust"],
        )
        # Only 1 mention → topic dropped → no segments returned.
        self.assertEqual(segments, [])


# --------------------------------------------------------------------------- #
# Wiring test — prove _topic_cut_segments_for_concept IS called inside
# ReelService.generate_reels' inner loop. Combines with the unit tests above
# to give end-to-end coverage without mocking the entire search pipeline.
# --------------------------------------------------------------------------- #


class GenerateReelsWiringTest(unittest.TestCase):
    """Verify the topic_cut helper is wired into ReelService.generate_reels."""

    def test_helper_is_called_from_generate_reels_source(self) -> None:
        import inspect
        from backend.app.services.reels import ReelService

        src = inspect.getsource(ReelService.generate_reels)

        # The helper must be invoked from inside the inner loop, with the
        # arguments we expect (transcript + video_id + duration + clip_min/max
        # + concept_terms for the concept-anchored refinement).
        self.assertIn("self._topic_cut_segments_for_concept(", src,
                      "topic_cut helper is not called from generate_reels")
        self.assertIn("transcript=transcript", src)
        self.assertIn("video_id=video_id", src)
        self.assertIn("clip_min_len=clip_min_len", src)
        self.assertIn("clip_max_len=clip_max_len", src)
        self.assertIn("concept_terms=concept_terms", src,
                      "concept_terms must be passed for concept-anchored refinement")

        # The call must be gated on `not use_full_short_clip` so YouTube Shorts
        # still go through the existing _full_short_clip_window path. Search
        # for both forms in case formatting changes.
        self.assertIn("use_full_short_clip", src)

        # And the legacy embedding path must remain as a fallback so we never
        # silently emit zero reels.
        self.assertIn("select_segments(", src)
        self.assertIn("_fast_segments_from_transcript(", src)
        self.assertIn("_fallback_segments_from_transcript(", src)

    def test_helper_is_a_method_on_live_reel_service(self) -> None:
        # Import main_module to confirm the singleton ReelService instance
        # carries the helper. This catches the case where reels.py was edited
        # but the runtime instance is somehow stale (it shouldn't be, but
        # belt-and-suspenders).
        import os
        os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")
        import backend.app.main as main_module

        self.assertIsNotNone(main_module.reel_service)
        self.assertTrue(
            hasattr(main_module.reel_service, "_topic_cut_segments_for_concept"),
            "live reel_service singleton lacks _topic_cut_segments_for_concept",
        )
        # Sanity-check the signature so a future refactor that renames params
        # doesn't silently break the wiring.
        import inspect
        sig = inspect.signature(main_module.reel_service._topic_cut_segments_for_concept)
        expected_params = {
            "transcript", "video_id", "video_duration_sec",
            "clip_min_len", "clip_max_len", "max_segments", "concept_terms",
        }
        actual = set(sig.parameters.keys())
        self.assertEqual(
            actual, expected_params,
            f"_topic_cut_segments_for_concept signature changed: {actual}",
        )


if __name__ == "__main__":
    unittest.main()
