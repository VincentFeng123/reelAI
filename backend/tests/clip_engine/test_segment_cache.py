import json
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
        "learning_objective": "Explain how forces change motion",
        "facet": "concept",
        "reason": "Explains the first lesson",
        "kind": "educational",
        "informativeness": 0.9,
        "topic_relevance": 0.8,
        "educational_importance": 0.8,
        "directly_teaches_topic": True,
        "substantive": True,
        "factually_grounded": True,
        "topic_evidence_quote": "The first lesson explains how forces change",
        "self_contained": True,
        "is_standalone": True,
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
    assert baseline.startswith("clip-segmentation:quality_silence_v31:v21:")

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
    assert _key(transcript, {"max_clips": 0}) == _key(transcript, {})
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
        assert _key(transcript, {**duration_policy, field: value}) == duration_key
    assert _key(transcript, {"segment_enrich_clips": True}) == _key(
        transcript,
        {"segment_enrich_clips": False},
    )
    assert _key(transcript, {"_segment_routing_mode": "flash_only"}) == baseline
    assert _key(transcript, {"_knowledge_level": "beginner"}) == _key(
        transcript, {"_knowledge_level": "advanced"},
    )

    original_primary = segment_cache.pipeline_config.SEGMENT_FLASH_MODEL
    monkeypatch.setattr(segment_cache.pipeline_config, "SEGMENT_FLASH_MODEL", "new-model")
    assert _key(transcript) != baseline
    monkeypatch.setattr(
        segment_cache.pipeline_config, "SEGMENT_FLASH_MODEL", original_primary,
    )
    monkeypatch.setattr(
        segment_cache.pipeline_config,
        "SEGMENT_FLASH_FALLBACK_MODEL",
        "new-fallback-model",
    )
    assert _key(transcript) != baseline


def test_segmenter_source_signature_includes_imported_validators(monkeypatch) -> None:
    from backend import gemini_client
    from backend.pipeline import discourse, sentences

    real_read_bytes = segment_cache.Path.read_bytes
    segment_cache._segmenter_source_signature.cache_clear()
    baseline = segment_cache._segmenter_source_signature()
    assert baseline is not None

    for module in (discourse, sentences, gemini_client):
        target_name = segment_cache.Path(module.__file__).name

        def changed_validator(path, *, expected_name=target_name):
            content = real_read_bytes(path)
            return (
                content + b"\n# changed imported validator"
                if path.name == expected_name
                else content
            )

        monkeypatch.setattr(segment_cache.Path, "read_bytes", changed_validator)
        segment_cache._segmenter_source_signature.cache_clear()
        changed = segment_cache._segmenter_source_signature()
        assert changed is not None
        assert changed != baseline

    segment_cache._segmenter_source_signature.cache_clear()


def test_segment_cache_revalidates_public_clip_contract() -> None:
    transcript = _transcript()
    settings = {"segment_accept_partial_flash": True}
    assert segment_cache._valid_clips(
        [_clip()], transcript=transcript, settings=settings
    ) == [_clip()]

    for field in (
        "informativeness",
        "topic_relevance",
        "educational_importance",
    ):
        below_floor = _clip()
        below_floor[field] = 0.74
        assert segment_cache._valid_clips(
            [below_floor], transcript=transcript, settings=settings
        ) is None

        at_floor = _clip()
        at_floor[field] = 0.75
        assert segment_cache._valid_clips(
            [at_floor], transcript=transcript, settings=settings
        ) == [at_floor]

    medium_uncertainty = _clip()
    medium_uncertainty.update({
        "uncertainty": "medium",
        "uncertainty_reasons": ["boundary_ambiguous"],
    })
    assert segment_cache._valid_clips(
        [medium_uncertainty], transcript=transcript, settings=settings
    ) == [medium_uncertainty]

    boundary_only_high_uncertainty = _clip()
    boundary_only_high_uncertainty.update({
        "uncertainty": "high",
        "uncertainty_reasons": ["boundary_ambiguous", "overlap_risk"],
    })
    assert segment_cache._valid_clips(
        [boundary_only_high_uncertainty], transcript=transcript, settings=settings
    ) == [boundary_only_high_uncertainty]

    for uncertainty_reason in ("topic_ambiguous", "incomplete_context"):
        content_high_uncertainty = _clip()
        content_high_uncertainty.update({
            "uncertainty": "high",
            "uncertainty_reasons": [uncertainty_reason],
        })
        assert segment_cache._valid_clips(
            [content_high_uncertainty], transcript=transcript, settings=settings
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


def test_segment_cache_keeps_distinct_facets_inside_one_coarse_cue() -> None:
    transcript = _transcript()
    first = _clip()
    first.update({
        "selection_candidate_id": "force-definition",
        "learning_objective": "Define force as an interaction",
        "facet": "force definition",
        "sequence_index": 1,
    })
    second = _clip()
    second.update({
        "selection_candidate_id": "motion-effect",
        "learning_objective": "Describe how motion changes",
        "facet": "acceleration effect",
        "sequence_index": 2,
    })

    assert segment_cache._valid_clips(
        [first, second], transcript=transcript, settings={}
    ) == [first, second]


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


def test_segment_cache_preserves_difficulty_order_not_chronology() -> None:
    transcript = _transcript()
    easy_later = _clip()
    easy_later.update({
        "start": 6.0,
        "end": 9.0,
        "difficulty": 0.2,
        "topic_evidence_quote": "The second lesson applies those forces to circular",
        "sequence_index": 1,
    })
    hard_earlier = _clip()
    hard_earlier.update({
        "difficulty": 0.8,
        "sequence_index": 2,
        "learning_objective": "Analyze force vectors quantitatively",
        "facet": "vector analysis",
    })

    assert segment_cache._valid_clips(
        [easy_later, hard_earlier], transcript=transcript, settings={}
    ) == [easy_later, hard_earlier]
    assert segment_cache._valid_clips(
        [hard_earlier, easy_later], transcript=transcript, settings={}
    ) is None


def test_segment_cache_accepts_more_than_sixteen_distinct_candidates() -> None:
    transcript = {
        "artifact_key": "transcript:exhaustive",
        "segments": [
            {
                "cue_id": f"c{index}",
                "start": index * 5.0,
                "end": (index + 1) * 5.0,
                "text": f"Lesson {index} explains forces and motion clearly for students.",
            }
            for index in range(20)
        ],
        "words": [],
        "duration": 100.0,
    }
    clips = []
    for index in range(20):
        clip = _clip()
        clip.update({
            "start": index * 5.0 + 0.5,
            "end": index * 5.0 + 4.5,
            "title": f"Lesson {index}",
            "learning_objective": f"Explain lesson {index}",
            "facet": f"lesson-{index}",
            "reason": f"Lesson {index} directly teaches motion",
            "topic_evidence_quote": (
                f"Lesson {index} explains forces and motion clearly"
            ),
            "sequence_index": index + 1,
        })
        clips.append(clip)

    assert segment_cache._valid_clips(
        clips, transcript=transcript, settings={}
    ) == clips


def test_segment_cache_accepts_complete_clips_longer_than_180_seconds() -> None:
    transcript = _transcript()
    transcript["segments"] = [{
        "cue_id": "long-cue",
        "start": 0.0,
        "end": 240.0,
        "text": "The first lesson explains how forces change an object's motion.",
    }]
    transcript["duration"] = 240.0
    clip = _clip()
    clip.update({"start": 0.0, "end": 240.0})

    assert segment_cache._valid_clips(
        [clip], transcript=transcript, settings={}
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
    cached_clip = _clip()
    cached_clip.update({
        "uncertainty": "high",
        "uncertainty_reasons": ["boundary_ambiguous", "overlap_risk"],
    })

    segment_cache.store_segment_result(
        cache_key,
        [cached_clip],
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
    ) == ([cached_clip], "cached")

    for uncertainty_reason in ("topic_ambiguous", "incomplete_context"):
        rejected_key = _key(
            transcript,
            settings,
            topic=f"rejected-{uncertainty_reason}",
        )
        rejected_clip = _clip()
        rejected_clip.update({
            "uncertainty": "high",
            "uncertainty_reasons": [uncertainty_reason],
        })
        segment_cache.store_segment_result(
            rejected_key,
            [rejected_clip],
            "must not cache",
            video_id=VIDEO_ID,
            transcript=transcript,
            settings=settings,
        )
        assert connection.execute(
            "SELECT 1 FROM llm_cache WHERE cache_key = ?", (rejected_key,)
        ).fetchone() is None
        assert segment_cache.load_segment_result(
            rejected_key,
            video_id=VIDEO_ID,
            transcript=transcript,
            settings=settings,
        ) is None

    stale_payload = json.loads(
        connection.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()[0]
    )
    stale_payload["selection_contract_version"] = "quality_silence_v30"
    connection.execute(
        "UPDATE llm_cache SET response_json = ? WHERE cache_key = ?",
        (json.dumps(stale_payload), cache_key),
    )
    assert segment_cache.load_segment_result(
        cache_key,
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    ) is None
    segment_cache.store_segment_result(
        cache_key,
        [cached_clip],
        "cached",
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    )

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


def test_segment_cache_round_trips_empty_qualifying_result(monkeypatch) -> None:
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

    segment_cache.store_segment_result(
        cache_key,
        [],
        "No qualifying clips.",
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    )

    assert segment_cache.load_segment_result(
        cache_key,
        video_id=VIDEO_ID,
        transcript=transcript,
        settings=settings,
    ) == ([], "No qualifying clips.")
    connection.close()
