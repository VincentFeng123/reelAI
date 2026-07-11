import importlib

import dotenv

from backend import config

def test_topic_engine_defaults():
    assert config.CLIP_ENGINE in ("gemini", "topic", "unit")
    assert config.CLIP_ENGINE == "gemini"
    assert config.SEGMENT_MIN_CLIP_S == 1.0
    assert config.SEGMENT_MAX_CLIP_S == 180.0
    assert config.SEGMENT_INFORMATIVENESS_MIN == 0.6
    assert config.SEGMENT_TOPIC_RELEVANCE_MIN == 0.6
    assert config.TOPIC_MAX_CLIPS == 40      # safety ceiling; ship all substantive topics (TreeSeg caps at 24)
    assert config.CLIP_MAX_S == 75.0
    assert config.CLIP_TARGET_S == 58.0
    assert config.TOPIC_INFORMATIVENESS_MIN == 0.5
    assert config.TOPIC_BOUNDARY_WINDOW == 3
    assert "clip_engine" in config.DEFAULTS and config.DEFAULTS["clip_engine"] is None


def test_authoring_model_is_flash_topic_model_is_pro():
    # bulk authoring stays on the fast model; only the new topic calls use the pro model
    assert config.GEMINI_MODEL == "gemini-2.5-flash"
    assert config.TOPIC_MODEL == "gemini-3.1-pro-preview"


def test_segment_router_defaults_are_fail_closed_and_pro_compatible():
    assert config.SEGMENT_ROUTING_MODE == "pro_only"
    assert config.SEGMENT_FLASH_MODEL == "gemini-3.5-flash"
    assert config.SEGMENT_PRO_MODEL == "gemini-3.1-pro-preview"
    assert config.SEGMENT_MODEL == config.SEGMENT_PRO_MODEL
    assert config.SEGMENT_HYBRID_PERCENT == 0.0


def _reload_segment_config(monkeypatch, **values):
    keys = {
        "SEGMENT_ROUTING_MODE", "SEGMENT_FLASH_MODEL", "SEGMENT_PRO_MODEL",
        "SEGMENT_MODEL", "SEGMENT_HYBRID_PERCENT",
    }
    with monkeypatch.context() as patch:
        patch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
        for key in keys:
            patch.delenv(key, raising=False)
        for key, value in values.items():
            patch.setenv(key, value)
        loaded = importlib.reload(config)
        result = {
            "mode": loaded.SEGMENT_ROUTING_MODE,
            "flash": loaded.SEGMENT_FLASH_MODEL,
            "pro": loaded.SEGMENT_PRO_MODEL,
            "legacy": loaded.SEGMENT_MODEL,
            "percent": loaded.SEGMENT_HYBRID_PERCENT,
        }
    importlib.reload(config)
    return result


def test_segment_router_rejects_invalid_values_and_clamps_percent(monkeypatch):
    invalid = _reload_segment_config(
        monkeypatch,
        SEGMENT_ROUTING_MODE="not-a-mode",
        SEGMENT_HYBRID_PERCENT="nan",
        SEGMENT_FLASH_MODEL="",
    )
    assert invalid == {
        "mode": "pro_only",
        "flash": "gemini-3.5-flash",
        "pro": "gemini-3.1-pro-preview",
        "legacy": "gemini-3.1-pro-preview",
        "percent": 0.0,
    }
    assert _reload_segment_config(
        monkeypatch, SEGMENT_HYBRID_PERCENT="101",
    )["percent"] == 100.0
    assert _reload_segment_config(
        monkeypatch, SEGMENT_HYBRID_PERCENT="-1",
    )["percent"] == 0.0


def test_legacy_pro_override_wins_then_explicit_pro_model_is_fallback(monkeypatch):
    legacy = _reload_segment_config(
        monkeypatch,
        SEGMENT_PRO_MODEL="gemini-3.1-pro-explicit",
        SEGMENT_MODEL="gemini-3.1-pro-legacy",
    )
    assert legacy["pro"] == legacy["legacy"] == "gemini-3.1-pro-legacy"

    explicit = _reload_segment_config(
        monkeypatch, SEGMENT_PRO_MODEL="gemini-3.1-pro-explicit",
    )
    assert explicit["pro"] == explicit["legacy"] == "gemini-3.1-pro-explicit"
