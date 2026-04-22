import unittest
from contextlib import contextmanager
from unittest import mock

from backend.tests import mass_clip_audit


@contextmanager
def _fake_conn():
    yield object()


class _FakeYouTubeService:
    def __init__(self, transcript, *, source_kind="yt_dlp_subtitle", exc=None):
        self.transcript = transcript
        self.source_kind = source_kind
        self.exc = exc
        self.calls = []

    def get_transcript(self, conn, video_id):
        self.calls.append(("get_transcript", video_id))
        if self.exc is not None:
            raise self.exc
        return self.transcript

    def get_transcript_quality(self, conn, video_id):
        self.calls.append(("get_transcript_quality", video_id))
        return {"source_kind": self.source_kind}


class MassClipAuditFetchTranscriptTests(unittest.TestCase):
    def test_fetch_transcript_uses_service_fallback_and_normalizes_rows(self) -> None:
        service = _FakeYouTubeService([
            {"start": 1.5, "end": 3.0, "text": "Hello world"},
            {"start": 3.0, "duration": 2.25, "text": "Next cue"},
            {"start": 9.0, "duration": 1.0, "text": ""},
        ])
        with mock.patch.object(mass_clip_audit, "get_conn", return_value=_fake_conn()):
            raw, kind = mass_clip_audit.fetch_transcript("abc123def45", youtube_service=service)

        self.assertEqual(kind, "yt_dlp_subtitle")
        self.assertEqual(
            raw,
            [
                {"text": "Hello world", "start": 1.5, "duration": 1.5},
                {"text": "Next cue", "start": 3.0, "duration": 2.25},
            ],
        )
        self.assertEqual(
            service.calls,
            [
                ("get_transcript", "abc123def45"),
                ("get_transcript_quality", "abc123def45"),
            ],
        )

    def test_fetch_transcript_returns_empty_on_service_failure(self) -> None:
        service = _FakeYouTubeService([], exc=RuntimeError("boom"))
        with mock.patch.object(mass_clip_audit, "get_conn", return_value=_fake_conn()):
            raw, kind = mass_clip_audit.fetch_transcript("abc123def45", youtube_service=service)

        self.assertEqual(raw, [])
        self.assertEqual(kind, "")


if __name__ == "__main__":
    unittest.main()
