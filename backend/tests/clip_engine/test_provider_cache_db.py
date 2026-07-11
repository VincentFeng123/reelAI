from datetime import datetime, timezone
from types import SimpleNamespace

from backend.app import db
from backend.app.clip_engine.provider_cache import (
    DatabaseProviderCache,
    TRANSCRIPT_SCHEMA_VERSION,
    TranscriptArtifact,
    search_cache_key,
    transcript_artifact_key,
)

VIDEO_ID = "dQw4w9WgXcQ"


def test_sqlite_provider_cache_round_trip_and_tombstone_filter(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        db,
        "get_settings",
        lambda: SimpleNamespace(data_dir=str(tmp_path), database_url=""),
    )
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    cache = DatabaseProviderCache()

    cache_key = search_cache_key(
        query="calculus", filters={"duration": "short"}, language="en", page_token=None
    )
    cache.put_search(
        cache_key,
        {"videos": [{"id": VIDEO_ID}], "next_page_token": None},
        {
            "normalized_query": "calculus",
            "filters": {"duration": "short"},
            "language": "en",
            "page_token": "",
        },
    )
    assert cache.get_search(cache_key).payload["videos"][0]["id"] == VIDEO_ID

    created_at = datetime.now(timezone.utc).isoformat()
    artifact = TranscriptArtifact(
        artifact_key=transcript_artifact_key(
            video_id=VIDEO_ID,
            provider="supadata",
            requested_language="en",
            returned_language="en",
            native_mode=True,
        ),
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        segments=[
            {"cue_id": "native-0", "start": 0.0, "end": 2.0, "text": "cue", "lang": "en"}
        ],
        duration_sec=2.0,
        created_at=created_at,
    )
    cache.put_transcript(artifact)
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) == artifact

    auto_artifact = TranscriptArtifact(
        artifact_key=transcript_artifact_key(
            video_id=VIDEO_ID,
            provider="supadata",
            requested_language="fr",
            returned_language="fr",
            native_mode=False,
        ),
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="fr",
        returned_language="fr",
        native_mode=False,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        segments=[
            {"cue_id": "auto-0", "start": 0.0, "end": 2.0, "text": "cue auto", "lang": "fr"}
        ],
        duration_sec=2.0,
        created_at=created_at,
    )
    cache.put_transcript(auto_artifact)
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="fr",
        native_mode=False,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) == auto_artifact
    with db.get_conn() as connection:
        stored = db.fetch_one(
            connection,
            "SELECT native_mode FROM transcript_artifacts WHERE cache_key = ?",
            (auto_artifact.artifact_key,),
        )
    assert int(stored["native_mode"]) == 0

    with db.get_conn(transactional=True) as connection:
        db.upsert(
            connection,
            "blocked_video_tombstones",
            {
                "video_id": VIDEO_ID,
                "canonical_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
                "source_url": "",
                "reason": "test",
                "created_at": created_at,
                "updated_at": created_at,
            },
            pk="video_id",
        )
    assert cache.get_search(cache_key).payload["videos"] == []
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) is None
