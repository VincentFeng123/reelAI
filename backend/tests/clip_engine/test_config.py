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


def test_segment_model_defaults_to_flash_with_pro_fallback(monkeypatch):
    monkeypatch.delenv("TOPIC_MODEL", raising=False)
    monkeypatch.delenv("SEGMENT_MODEL", raising=False)
    monkeypatch.delenv("SEGMENT_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.GEMINI_MODEL == "gemini-3.5-flash"
    assert cfg.TOPIC_MODEL == "gemini-3.1-pro-preview"
    assert cfg.SEGMENT_MODEL == cfg.GEMINI_MODEL
    assert cfg.SEGMENT_FALLBACK_MODEL == cfg.TOPIC_MODEL


def test_assessment_model_stays_flash_when_shared_or_assessment_models_are_pro(
    monkeypatch,
):
    with monkeypatch.context() as patch:
        patch.setenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
        patch.setenv("SEGMENT_MODEL", "gemini-3.1-pro-preview")
        patch.delenv("ASSESSMENT_MODEL", raising=False)
        cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
        assert cfg.ASSESSMENT_MODEL == "gemini-3.5-flash"

        patch.setenv("ASSESSMENT_MODEL", "gemini-3.1-pro-preview")
        cfg = importlib.reload(cfg)
        assert cfg.ASSESSMENT_MODEL == "gemini-3.5-flash"

        patch.setenv("ASSESSMENT_MODEL", "gemini-3.1-flash-lite")
        cfg = importlib.reload(cfg)
        assert cfg.ASSESSMENT_MODEL == "gemini-3.1-flash-lite"

    importlib.reload(cfg)


def test_lesson_order_model_never_consumes_pro_selector_budget(monkeypatch):
    with monkeypatch.context() as patch:
        patch.delenv("LESSON_ORDER_MODEL", raising=False)
        cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
        assert cfg.LESSON_ORDER_MODEL == "gemini-2.5-flash-lite"

        patch.setenv("LESSON_ORDER_MODEL", "gemini-3.1-pro-preview")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_MODEL == "gemini-2.5-flash-lite"

        patch.setenv("LESSON_ORDER_MODEL", "gemini-3.1-flash-lite")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_MODEL == "gemini-2.5-flash-lite"

        patch.setenv("LESSON_ORDER_MODEL", "gemini-2.5-flash")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_MODEL == "gemini-2.5-flash"

    importlib.reload(cfg)


def test_lesson_order_retry_uses_a_distinct_non_pro_25_model(monkeypatch):
    with monkeypatch.context() as patch:
        patch.delenv("LESSON_ORDER_MODEL", raising=False)
        cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
        assert cfg.LESSON_ORDER_MODEL == "gemini-2.5-flash-lite"
        assert cfg.LESSON_ORDER_FALLBACK_MODEL == "gemini-2.5-flash"

        patch.setenv("LESSON_ORDER_MODEL", "gemini-3.1-pro-preview")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_FALLBACK_MODEL == "gemini-2.5-flash"

        patch.setenv("LESSON_ORDER_MODEL", "gemini-2.5-flash")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_FALLBACK_MODEL == "gemini-2.5-flash-lite"

        patch.setenv("LESSON_ORDER_MODEL", "models/gemini-2.5-flash")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_FALLBACK_MODEL == "gemini-2.5-flash-lite"

        patch.setenv("LESSON_ORDER_MODEL", "gemini-2.5-flash-001")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_FALLBACK_MODEL == "gemini-2.5-flash-lite"

        patch.setenv("LESSON_ORDER_MODEL", "gemini-2.5-flash-lite-001")
        cfg = importlib.reload(cfg)
        assert cfg.LESSON_ORDER_FALLBACK_MODEL == "gemini-2.5-flash"

    importlib.reload(cfg)


def test_curation_gate_defaults(monkeypatch):
    monkeypatch.delenv("SEGMENT_MAX_CLIP_S", raising=False)
    monkeypatch.delenv("SEGMENT_INFORMATIVENESS_MIN", raising=False)
    monkeypatch.delenv("SEGMENT_TOPIC_RELEVANCE_MIN", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.SEGMENT_MAX_CLIP_S == 180.0
    assert cfg.SEGMENT_INFORMATIVENESS_MIN == 0.6
    assert cfg.SEGMENT_TOPIC_RELEVANCE_MIN == 0.6


def test_search_breadth_uses_fast_coverage_default(monkeypatch):
    monkeypatch.delenv("CLIP_SEARCH_BREADTH", raising=False)
    cfg = importlib.reload(importlib.import_module("backend.app.clip_engine.config"))
    assert cfg.SEARCH_BREADTH == 3


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
    from backend.app.clip_engine.errors import ProviderConfigurationError
    with pytest.raises(ProviderConfigurationError) as exc_info:
        cfg.require_supadata_key()
    assert exc_info.value.as_dict()["code"] == "provider_configuration"
