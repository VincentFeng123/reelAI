from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VERCEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services import importance_ranker, topic_cut  # noqa: E402
from backend.app.services.topic_cut import TopicReel, TranscriptCue, _filter_reels_by_query  # noqa: E402


def _cue(start: float, duration: float, text: str) -> TranscriptCue:
    return TranscriptCue(start=start, duration=duration, text=text)


def _reel(start: float, end: float, label: str, summary: str = "") -> TopicReel:
    return TopicReel(
        video_id="vid-1",
        t_start=start,
        t_end=end,
        label=label,
        summary=summary,
    )


def _physics_fixture() -> tuple[list[TopicReel], list[TranscriptCue]]:
    reels = [
        _reel(0.0, 40.0, "Physics intro"),
        _reel(40.0, 80.0, "Physics rotation"),
        _reel(80.0, 120.0, "Physics motion"),
        _reel(120.0, 160.0, "Physics sponsor"),
        _reel(160.0, 200.0, "Physics energy"),
        _reel(200.0, 240.0, "Physics momentum"),
    ]
    cues = [
        _cue(0.0, 40.0, "Hey everyone, welcome back to the channel. Today we are starting our physics review."),
        _cue(40.0, 40.0, "Torque equals force times lever arm. This is the core rotational physics idea."),
        _cue(80.0, 40.0, "Angular acceleration depends on torque and moment of inertia in rotational physics."),
        _cue(120.0, 40.0, "This video is sponsored by Brilliant. Use code TORQUE for a discount."),
        _cue(160.0, 40.0, "Conservation of energy explains how potential becomes kinetic energy in physics."),
        _cue(200.0, 40.0, "Momentum compares impulse and change in velocity in physics problems."),
    ]
    return reels, cues


def _trim_fixture() -> tuple[list[TopicReel], list[TranscriptCue]]:
    reels = [
        _reel(0.0, 50.0, "Gradient descent"),
    ]
    cues = [
        _cue(0.0, 10.0, "Hey everyone, welcome back to the channel."),
        _cue(10.0, 10.0, "Gradient descent moves in the direction of steepest decrease."),
        _cue(20.0, 10.0, "The negative gradient identifies the most important direction to follow."),
        _cue(30.0, 10.0, "Each update subtracts the learning rate times the gradient."),
        _cue(40.0, 10.0, "Thanks for watching and subscribe for more."),
    ]
    return reels, cues


def _reel_payloads(reels: list[TopicReel]) -> list[dict[str, object]]:
    return [reel.to_dict() for reel in reels]


class TopicCutImportanceTests(unittest.TestCase):
    def test_flag_off_parity(self) -> None:
        reels, cues = _physics_fixture()

        with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("REELS_IMPORTANCE_RANKER_ENABLED", None)
                baseline = _filter_reels_by_query(reels, "torque", cues=cues, video_duration_sec=240.0)
            with mock.patch.dict(
                os.environ,
                {"REELS_IMPORTANCE_RANKER_ENABLED": "0"},
                clear=False,
            ):
                disabled = _filter_reels_by_query(reels, "torque", cues=cues, video_duration_sec=240.0)

        self.assertEqual(_reel_payloads(baseline), _reel_payloads(disabled))

    def test_sponsor_intro_recap_outro_reels_are_excluded(self) -> None:
        reels, cues = _physics_fixture()
        recap_reel = _reel(240.0, 280.0, "Physics recap")
        recap_cue = _cue(240.0, 40.0, "Let's recap what we covered earlier before moving on.")
        reels.append(recap_reel)
        cues.append(recap_cue)

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
                kept = _filter_reels_by_query(
                    reels,
                    "physics review",
                    cues=cues,
                    video_duration_sec=280.0,
                )

        labels = {reel.label.lower() for reel in kept}
        self.assertNotIn("physics intro", labels)
        self.assertNotIn("physics sponsor", labels)
        self.assertNotIn("physics recap", labels)

    def test_narrow_query_returns_at_most_two_reels(self) -> None:
        reels, cues = _physics_fixture()

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
                kept = _filter_reels_by_query(
                    reels,
                    "torque",
                    cues=cues,
                    video_duration_sec=240.0,
                )

        self.assertLessEqual(len(kept), 2)

    def test_broad_query_returns_three_to_five_reels(self) -> None:
        reels, cues = _physics_fixture()

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
                kept = _filter_reels_by_query(
                    reels,
                    "physics review",
                    cues=cues,
                    video_duration_sec=240.0,
                )

        self.assertGreaterEqual(len(kept), 3)
        self.assertLessEqual(len(kept), 5)

    def test_ranker_called_once_when_flag_on(self) -> None:
        reels, cues = _physics_fixture()

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
                with mock.patch(
                    "backend.app.services.importance_ranker.rank_segments",
                    wraps=importance_ranker.rank_segments,
                ) as rank_mock:
                    _filter_reels_by_query(
                        reels,
                        "torque",
                        cues=cues,
                        video_duration_sec=240.0,
                    )

        self.assertEqual(rank_mock.call_count, 1)

    def test_structural_opener_is_trimmed_without_dropping_reel(self) -> None:
        reels, cues = _trim_fixture()

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
                kept = _filter_reels_by_query(
                    reels,
                    "gradient descent",
                    cues=cues,
                    video_duration_sec=50.0,
                )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].t_start, 10.0)
        self.assertEqual(kept[0].cue_start_idx, 1)

    def test_structural_tail_is_trimmed_and_ends_on_punctuation(self) -> None:
        reels, cues = _trim_fixture()

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
                kept = _filter_reels_by_query(
                    reels,
                    "gradient descent",
                    cues=cues,
                    video_duration_sec=50.0,
                )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].t_end, 40.0)
        kept_text = " ".join(
            cue.text
            for cue in cues
            if cue.start >= kept[0].t_start and cue.end <= kept[0].t_end
        )
        self.assertTrue(kept_text.strip().endswith("."))

    def test_trim_preserves_existing_end_when_only_opener_is_removed(self) -> None:
        reels = [
            TopicReel(
                video_id="vid-1",
                t_start=0.0,
                t_end=40.37,
                label="Gradient descent",
            )
        ]
        cues = [
            _cue(0.0, 10.0, "Hey everyone, welcome back to the channel."),
            _cue(10.0, 10.0, "Gradient descent moves in the direction of steepest decrease."),
            _cue(20.0, 10.0, "The negative gradient identifies the most important direction to follow."),
            _cue(30.0, 10.37, "Each update subtracts the learning rate times the gradient."),
        ]

        with mock.patch.dict(os.environ, {"REELS_IMPORTANCE_RANKER_ENABLED": "1"}, clear=False):
            with mock.patch("backend.app.services.topic_cut._compute_semantic_scores", return_value=None):
                kept = _filter_reels_by_query(
                    reels,
                    "gradient descent",
                    cues=cues,
                    video_duration_sec=40.37,
                )

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].t_start, 10.0)
        self.assertEqual(kept[0].t_end, 40.37)


if __name__ == "__main__":
    unittest.main()
