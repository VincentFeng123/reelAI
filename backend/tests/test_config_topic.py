from backend import config

def test_topic_engine_defaults():
    assert config.CLIP_ENGINE in ("topic", "unit")
    assert config.CLIP_ENGINE == "topic"          # default routes to the new engine
    assert config.TOPIC_MAX_CLIPS == 10
    assert config.CLIP_MAX_S == 75.0
    assert config.CLIP_TARGET_S == 58.0
    assert config.TOPIC_INFORMATIVENESS_MIN == 0.5
    assert config.TOPIC_BOUNDARY_WINDOW == 3
    assert "clip_engine" in config.DEFAULTS and config.DEFAULTS["clip_engine"] is None
