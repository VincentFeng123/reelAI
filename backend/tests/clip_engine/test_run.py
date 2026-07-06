# backend/tests/clip_engine/test_run.py
import pytest
from backend.app.clip_engine import run
from backend.app.clip_engine.errors import UnsupportedURLError


def test_rejects_non_youtube():
    with pytest.raises(UnsupportedURLError):
        run.clip("https://vimeo.com/123", "topic")


def test_clip_builds_embed_urls(monkeypatch):
    fake_tx = {"segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}],
               "words": [], "duration": 5.0}
    monkeypatch.setattr(run, "_transcribe", lambda url, video_id, settings: fake_tx)
    monkeypatch.setattr(run.gemini_segment, "segment_clips",
                        lambda transcript, settings, progress=None: (
                            [{"start": 1.0, "end": 4.0, "cut_end": 4.15, "title": "Bit",
                              "facet": "concept", "reason": "", "sequence_index": 1}], "1 clip"))
    out = run.clip("https://www.youtube.com/watch?v=abc123", "topic")
    assert out["video_id"] == "abc123"
    assert out["clips"][0]["embed_url"] == "https://www.youtube.com/embed/abc123?start=1&end=4&rel=0"
