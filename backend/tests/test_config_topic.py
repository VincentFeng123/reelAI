import importlib

import dotenv
import pytest

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


def test_segment_router_defaults_to_normal_flash_only_for_clip_selection():
    assert config.SEGMENT_ROUTING_MODE == "flash_only"
    assert config.SEGMENT_FLASH_MODEL == "gemini-3.5-flash"
    assert config.SEGMENT_FLASH_FALLBACK_MODEL == ""
    assert config.SEGMENT_PRO_MODEL == "gemini-3.1-pro-preview"
    assert config.SEGMENT_MODEL == config.SEGMENT_PRO_MODEL
    assert config.SEGMENT_HYBRID_PERCENT == 100.0


def _reload_segment_config(monkeypatch, **values):
    keys = {
        "SEGMENT_ROUTING_MODE", "SEGMENT_FLASH_MODEL",
        "SEGMENT_FLASH_FALLBACK_MODEL", "SEGMENT_PRO_MODEL", "SEGMENT_MODEL",
        "SEGMENT_HYBRID_PERCENT",
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
            "flash_fallback": loaded.SEGMENT_FLASH_FALLBACK_MODEL,
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
        "mode": "flash_only",
        "flash": "gemini-3.5-flash",
        "flash_fallback": "",
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


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "not-a-model",
    ],
)
def test_authoritative_flash_model_is_pinned_against_environment_overrides(
    monkeypatch,
    model,
):
    loaded = _reload_segment_config(
        monkeypatch,
        SEGMENT_FLASH_MODEL=model,
    )
    assert loaded["flash"] == "gemini-3.5-flash"


def test_explicit_pro_override_wins_then_legacy_pro_model_is_fallback(monkeypatch):
    explicit = _reload_segment_config(
        monkeypatch,
        SEGMENT_PRO_MODEL="gemini-3.1-pro-explicit",
        SEGMENT_MODEL="gemini-3.1-pro-legacy",
    )
    assert explicit["pro"] == explicit["legacy"] == "gemini-3.1-pro-explicit"

    legacy = _reload_segment_config(
        monkeypatch, SEGMENT_MODEL="gemini-3.1-pro-legacy",
    )
    assert legacy["pro"] == legacy["legacy"] == "gemini-3.1-pro-legacy"


@pytest.mark.parametrize("variable", ["SEGMENT_PRO_MODEL", "SEGMENT_MODEL"])
def test_non_pro_selector_override_cannot_downgrade_authoritative_selection(
    monkeypatch,
    variable,
):
    loaded = _reload_segment_config(monkeypatch, **{variable: "gemini-3.5-flash"})
    assert loaded["pro"] == loaded["legacy"] == "gemini-3.1-pro-preview"


def test_invalid_explicit_selector_can_use_a_valid_legacy_pro_fallback(monkeypatch):
    loaded = _reload_segment_config(
        monkeypatch,
        SEGMENT_PRO_MODEL="gemini-flash-proxy",
        SEGMENT_MODEL="gemini-3.1-pro-legacy",
    )
    assert loaded["pro"] == loaded["legacy"] == "gemini-3.1-pro-legacy"


def test_segment_flash_failover_model_is_configurable_and_can_be_disabled(monkeypatch):
    assert _reload_segment_config(
        monkeypatch,
        SEGMENT_FLASH_FALLBACK_MODEL="gemini-3.2-flash-lite",
    )["flash_fallback"] == "gemini-3.2-flash-lite"
    assert _reload_segment_config(
        monkeypatch,
        SEGMENT_FLASH_FALLBACK_MODEL="",
    )["flash_fallback"] == ""
