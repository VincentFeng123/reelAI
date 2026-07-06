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


def test_require_supadata_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    from backend.app.clip_engine.errors import SearchError
    with pytest.raises(SearchError):
        cfg.require_supadata_key()
