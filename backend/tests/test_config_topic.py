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
