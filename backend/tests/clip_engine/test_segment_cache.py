import sqlite3
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from backend.app.clip_engine import segment_cache


VIDEO_ID = "dQw4w9WgXcQ"


def _transcript() -> dict:
    return {
        "artifact_key": "transcript:v1",
        "segments": [
            {
                "cue_id": "c0",
                "start": 0.0,
                "end": 5.0,
                "text": "The first lesson explains how forces change an object's motion.",
            },
            {
                "cue_id": "c1",
                "start": 5.0,
                "end": 10.0,
                "text": "The second lesson applies those forces to circular motion.",
            },
        ],
        "words": [],
        "duration": 10.0,
    }


def _clip() -> dict:
    return {
        "start": 1.0,
        "end": 4.0,
        "title": "First lesson",
        "facet": "concept",
        "reason": "Explains the first lesson",
        "kind": "educational",
        "informativeness": 0.9,
        "topic_relevance": 0.8,
        "educational_importance": 0.7,
        "directly_teaches_topic": True,
        "substantive": True,
        "factually_grounded": True,
        "topic_evidence_quote": "The first lesson explains how forces change",
        "self_contained": True,
        "difficulty": 0.4,
        "summary": "",
        "takeaways": [],
        "match_reason": "",
        "assessment": None,
        "sequence_index": 1,
    }


def _key(transcript: dict, settings: dict | None = None, *, topic: str = "physics") -> str:
    return segment_cache.segment_cache_key(
        video_id=VIDEO_ID,
        topic=topic,
        transcript=transcript,
        settings=settings or {"segment_accept_partial_flash": True},
    )


def test_segment_cache_key_tracks_transcript_topic_and_policy(monkeypatch) -> None:
    transcript = _transcript()
    baseline = _key(transcript)
    assert baseline.startswith("clip-segmentation:v3:")

    changed_text = deepcopy(transcript)
    changed_text["segments"][0]["text"] = "changed lesson"
    changed_timing = deepcopy(transcript)
    changed_timing["segments"][0]["end"] = 4.5
    changed_duration = deepcopy(transcript)
    changed_duration["duration"] = 11.0

    assert _key(changed_text) != baseline
    assert _key(changed_timing) != baseline
    assert _key(changed_duration) != baseline
    assert _key(transcript, topic="Physics") != baseline
    assert _key(transcript, {"segment_accept_partial_flash": False}) == baseline
    assert _key(transcript, {"max_clips": 0}) != _key(transcript, {})
    duration_policy = {
        "_segment_target_sec": 38,
        "_segment_target_min_sec": 20,
        "_segment_target_max_sec": 55,
    }
    duration_key = _key(transcript, duration_policy)
    for field, value in (
        ("_segment_target_sec", 39),
        ("_segment_target_min_sec", 21),
        ("_segment_target_max_sec", 56),
    ):
        assert _key(transcript, {**duration_policy, field: value}) != duration_key
    assert _key(transcript, {"segment_enrich_clips": True}) != _key(
        transcript,
        {"segment_enrich_clips": False},
    )
    assert _key(transcript, {"_segment_routing_mode": "flash_only"}) != baseline

    monkeypatch.setattr(segment_cache.pipeline_config, "SEGMENT_FLASH_MODEL", "new-model")
    assert _key(transcript) != baseline


def test_segment_cache_revalidates_public_clip_contract() -> None:
    transcript = _transcript()
    settings = {"segment_accept_partial_flash": True}
    assert segment_cache._valid_clips(
        [_clip()], transcript=transcript, settings=settings
    ) == [_clip()]

    low_scored = _clip()
    low_scored.update({
        "informativeness": 0.0,
        "topic_relevance": 0.0,
        "educational_importance": 0.0,
        "difficulty": 0.0,
    })
    assert segment_cache._valid_clips(
        [low_scored], transcript=transcript, settings=settings
    ) == [low_scored]

    medium_uncertainty = _clip()
    medium_uncertainty.update({
        "uncertainty": "medium",
        "uncertainty_reasons": ["boundary_ambiguous"],
    })
    assert segment_cache._valid_clips(
        [medium_uncertainty], transcript=transcript, settings=settings
    ) == [medium_uncertainty]

    high_uncertainty = _clip()
    high_uncertainty.update({
        "uncertainty": "high",
        "uncertainty_reasons": ["incomplete_context"],
    })
    assert segment_cache._valid_clips(
        [high_uncertainty], transcript=transcript, settings=settings
    ) is None

    invalid = _clip()
    invalid["self_contained"] = False
    assert segment_cache._valid_clips(
        [invalid], transcript=transcript, settings=settings
    ) is None

    invalid = _clip()
    invalid["end"] = 11.0
    assert segment_cache._valid_clips(
        [invalid], transcript=transcript, settings=settings
    ) is None

    invalid = _clip()
    invalid["educational_importance"] = 1.01
    assert segment_cache._valid_clips(
        [invalid], transcript=transcript, settings=settings
    ) is None

    duplicate = _clip()
    duplicate["start"] = 1.5
    duplicate["end"] = 4.0
    duplicate["sequence_index"] = 2
    assert segment_cache._valid_clips(
        [_clip(), duplicate], transcript=transcript, settings=settings
    ) is None


def test_segment_cache_treats_requested_max_as_preference() -> None:
    transcript = _transcript()
    transcript["segments"] = [{
        "cue_id": "long-cue",
        "start": 0.0,
        "end": 70.0,
        "text": "The first lesson explains how forces change an object's motion.",
    }]
    transcript["duration"] = 70.0
    clip = _clip()
    clip.update({"start": 0.0, "end": 60.0})

    assert segment_cache._valid_clips(
        [clip],
        transcript=transcript,
        settings={"_segment_target_max_sec": 55},
    ) == [clip]
    assert segment_cache._valid_clips(
        [clip],
        transcript=transcript,
        settings={"_segment_target_max_sec": 60},
    ) == [clip]


def test_segment_cache_disables_shadow_and_unversioned_releases(monkeypatch) -> None:
    monkeypatch.setattr(segment_cache.pipeline_config, "SEGMENT_ROUTING_MODE", "shadow")
    assert segment_cache.cache_enabled() is False

    monkeypatch.setattr(segment_cache.pipeline_config, "SEGMENT_ROUTING_MODE", "hybrid")
    monkeypatch.setattr(segment_cache, "_segmenter_source_signature", lambda: None)
    assert segment_cache.cache_enabled() is False


def test_segment_cache_sqlite_round_trip_expiry_and_tombstone(monkeypatch) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE llm_cache (cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE blocked_video_tombstones (video_id TEXT PRIMARY KEY)"
    )

    @contextmanager
    def fake_get_conn(*, transactional: bool = False):
        try:
            yield connection
        except Exception:
            if transactional:
                connection.rollback()
            raise
        else:
            if transactional:
                connection.commit()

    monkeypatch.setattr(segment_cache, "get_conn", fake_get_conn)
    transcript = _transcript()
    settings = {"segment_accept_partial_flash": True}
    cache_key = _key(transcript, settings)
    low_scored = _clip()
    low_scored.update({
        "informativeness": 0.0,
        "topic_relevance": 0.0,
        "educational_importance": 0.0,
        "difficulty": 0.0,
    })

    segment_cache.store_segment_result(
        cache_key,
        [low_scored],
        "cached",
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    )
    assert segment_cache.load_segment_result(
        cache_key,
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    ) == ([low_scored], "cached")

    expired = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    connection.execute(
        "UPDATE llm_cache SET created_at = ? WHERE cache_key = ?",
        (expired, cache_key),
    )
    assert segment_cache.load_segment_result(
        cache_key,
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    ) is None

    segment_cache.store_segment_result(
        cache_key,
        [_clip()],
        "cached",
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    )
    connection.execute(
        "INSERT INTO blocked_video_tombstones (video_id) VALUES (?)",
        (VIDEO_ID,),
    )
    assert segment_cache.load_segment_result(
        cache_key,
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    ) is None
    connection.close()
