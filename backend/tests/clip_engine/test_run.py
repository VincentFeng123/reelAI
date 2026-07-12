# backend/tests/clip_engine/test_run.py
import pytest
from backend.app.clip_engine import run
from backend.app.clip_engine.errors import (
    ProviderAuthenticationError,
    ProviderQuotaError,
    UnsupportedURLError,
)


def test_rejects_non_youtube():
    with pytest.raises(UnsupportedURLError):
        run.clip("https://vimeo.com/123", "topic")


def test_clip_builds_embed_urls_and_forwards_topic(monkeypatch):
    fake_tx = {"segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}],
               "words": [], "duration": 5.0}
    seen: dict = {}

    def fake_segment(transcript, settings, progress=None, topic="", video_id=""):
        seen["topic"] = topic
        seen["video_id"] = video_id
        seen["accept_partial_flash"] = settings.get("segment_accept_partial_flash")
        return ([{"start": 1.0, "end": 4.0, "cut_end": 4.15, "title": "Bit",
                  "facet": "concept", "reason": "", "sequence_index": 1}], "1 clip")

    def fake_transcribe(url, video_id, settings):
        seen["transcript_url"] = url
        return fake_tx

    monkeypatch.setattr(run, "_transcribe", fake_transcribe)
    monkeypatch.setattr(run.gemini_segment, "segment_clips", fake_segment)
    out = run.clip(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&redirect=https://evil.test",
        "topic",
    )
    assert out["video_id"] == "dQw4w9WgXcQ"
    assert out["clips"][0]["embed_url"] == "https://www.youtube.com/embed/dQw4w9WgXcQ?start=1&end=4&rel=0"
    assert seen["topic"] == "topic"
    assert seen["video_id"] == "dQw4w9WgXcQ"
    assert seen["accept_partial_flash"] is True
    assert seen["transcript_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_live_runner_uses_the_canonical_practice_segmenter():
    assert run.gemini_segment.__name__ == "backend.pipeline.gemini_segment"


def test_transcript_provider_error_is_re_raised_unchanged(monkeypatch):
    from backend.app.clip_engine.clipper.pipeline import transcribe

    error = ProviderQuotaError(
        "quota", provider="supadata", operation="transcript", status_code=402
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(transcribe, "transcribe_supadata", fail)
    with pytest.raises(ProviderQuotaError) as exc_info:
        run._transcribe(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
            {},
        )
    assert exc_info.value is error


def test_segmentation_provider_error_is_re_raised_unchanged(monkeypatch):
    transcript = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "native cue"}],
        "words": [],
    }
    error = ProviderAuthenticationError(
        "auth", provider="gemini", operation="segmentation", status_code=403
    )
    monkeypatch.setattr(run, "_transcribe", lambda *args, **kwargs: transcript)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(run.gemini_segment, "segment_clips", fail)
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        run.clip("https://youtu.be/dQw4w9WgXcQ", "topic")
    assert exc_info.value is error
