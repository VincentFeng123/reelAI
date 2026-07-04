from backend import config


def test_short_instagram_targets():
    assert config.DEFAULTS["target_clip_duration_s"] == 45.0   # scoring aim (30-60s)
    assert config.DEFAULTS["min_clip_duration_s"] <= 15.0
    # max_clip_duration_s is the HARD overflow ceiling / ship cap (NOT a soft 90 cut): shorter
    # than the old 240 but well ABOVE the soft closure budget so onset-overflow has headroom.
    assert config.DEFAULTS["max_clip_duration_s"] == 180.0


def test_soft_budget_below_hard_ceiling():
    # the onset-overflow window (soft, hard] must be NON-EMPTY, else force-inline is inert
    # (Task 6 review Critical). The value that actually reaches build_candidate is
    # DEFAULTS["closure_max_span_s"] (a DEFAULTS entry, NOT the module constant) — guard BOTH
    # so the feature cannot silently re-inert if someone raises the soft budget later.
    assert config.DEFAULTS["closure_max_span_s"] < config.DEFAULTS["max_clip_duration_s"]
    assert config.CLOSURE_MAX_SPAN_S < config.DEFAULTS["max_clip_duration_s"]


def test_fewer_clips_by_default():
    assert config.MAX_SEGMENTS <= 8
