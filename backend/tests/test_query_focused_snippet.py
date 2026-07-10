from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ingestion.models import IngestTranscriptCue  # noqa: E402
from backend.app.ingestion.segment import snippet_for_window  # noqa: E402


def _cue(start: float, end: float, text: str) -> IngestTranscriptCue:
    return IngestTranscriptCue(
        start=start,
        end=end,
        text=text,
        words=[],
        word_source="native_caption",
    )


class QueryFocusedSnippetTests(unittest.TestCase):
    def test_focus_query_prefers_query_bearing_excerpt(self) -> None:
        cues = [
            _cue(0.0, 3.0, "Welcome back everyone and thanks for watching."),
            _cue(3.0, 7.0, "The gradient vector tells you which changes matter most."),
            _cue(7.0, 11.0, "That relative importance is what lets gradient descent work."),
            _cue(11.0, 14.0, "We'll cover more examples in a moment."),
        ]

        snippet = snippet_for_window(
            cues,
            0.0,
            14.0,
            max_chars=160,
            focus_query="gradient vector importance",
        )

        self.assertIn("gradient vector", snippet.lower())
        self.assertIn("importance", snippet.lower())
        self.assertNotIn("thanks for watching", snippet.lower())


if __name__ == "__main__":
    unittest.main()
