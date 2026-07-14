from backend.app.models import CommunitySettingsPayload


def test_deprecated_clip_durations_accept_out_of_range_and_inverted_values():
    payload = CommunitySettingsPayload.model_validate({
        "target_clip_duration_sec": 9_999,
        "target_clip_duration_min_sec": 500,
        "target_clip_duration_max_sec": -5,
    })

    assert payload.target_clip_duration_sec == 9_999
    assert payload.target_clip_duration_min_sec == 500
    assert payload.target_clip_duration_max_sec == -5


def test_deprecated_clip_durations_keep_legacy_and_nullable_decoding():
    legacy = CommunitySettingsPayload.model_validate_json(
        '{"target_clip_duration_sec":55,'
        '"target_clip_duration_min_sec":20,'
        '"target_clip_duration_max_sec":55}'
    )
    omitted = CommunitySettingsPayload()

    assert (
        legacy.target_clip_duration_sec,
        legacy.target_clip_duration_min_sec,
        legacy.target_clip_duration_max_sec,
    ) == (55, 20, 55)
    assert omitted.target_clip_duration_sec is None
    assert omitted.target_clip_duration_min_sec is None
    assert omitted.target_clip_duration_max_sec is None
