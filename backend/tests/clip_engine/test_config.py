import importlib
import pytest


def test_defaults_are_gemini_embed(monkeypatch):
    monkeypatch.delenv("CLIP_ENGINE", raising=False)
    monkeypatch.delenv("PRECISE_BOUNDARIES", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.CLIP_ENGINE == "gemini"
    assert cfg.OUTPUT_MODE == "embed"
    assert cfg.PRECISE_BOUNDARIES is False
    assert cfg.SEGMENT_FINE_SNAP is True
    assert cfg.CLIP_SEARCH_MAX_VIDEOS == 5


def test_segment_model_defaults_to_pro_tier(monkeypatch):
    """Segmentation must fall back to the PRO tier (TOPIC_MODEL), never to the
    cheap GEMINI_MODEL — the silent flash downgrade was a root cause of the
    bad-reels regression."""
    monkeypatch.delenv("TOPIC_MODEL", raising=False)
    monkeypatch.delenv("SEGMENT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.TOPIC_MODEL == "gemini-3.1-pro-preview"
    assert cfg.SEGMENT_MODEL == cfg.TOPIC_MODEL
    assert cfg.SEGMENT_MODEL != cfg.GEMINI_MODEL


def test_curation_gate_defaults(monkeypatch):
    monkeypatch.delenv("SEGMENT_MAX_CLIP_S", raising=False)
    monkeypatch.delenv("SEGMENT_INFORMATIVENESS_MIN", raising=False)
    monkeypatch.delenv("SEGMENT_TOPIC_RELEVANCE_MIN", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.SEGMENT_MAX_CLIP_S == 180.0
    assert cfg.SEGMENT_INFORMATIVENESS_MIN == 0.6
    assert cfg.SEGMENT_TOPIC_RELEVANCE_MIN == 0.6


def test_search_breadth_matches_adaptive_consensus_ceiling(monkeypatch):
    monkeypatch.delenv("CLIP_SEARCH_BREADTH", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.SEARCH_BREADTH == 6


def test_extract_video_id_accepts_v_param_anywhere():
    from backend.app.clip_engine.metadata import extract_video_id

    assert extract_video_id(
        "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ"
    ) == "dQw4w9WgXcQ"
    assert extract_video_id(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ) == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://vimeo.com/123") is None


def test_require_supadata_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    from backend.app.clip_engine.errors import SearchError
    with pytest.raises(SearchError):
        cfg.require_supadata_key()
