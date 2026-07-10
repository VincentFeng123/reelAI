from datetime import datetime, timedelta, timezone

import pytest

from backend.app.clip_engine.errors import ProviderBudgetExceededError
from backend.app.clip_engine.provider_cache import (
    MemoryProviderCache,
    TRANSCRIPT_SCHEMA_VERSION,
    TranscriptArtifact,
    normalize_filters,
    search_cache_key,
    transcript_artifact_key,
    validate_transcript_payload,
)
from backend.app.clip_engine.provider_runtime import (
    GenerationBudget,
    GenerationContext,
    bounded_retry_after,
)

VIDEO_ID = "dQw4w9WgXcQ"


def _artifact() -> TranscriptArtifact:
    created_at = datetime.now(timezone.utc).isoformat()
    key = transcript_artifact_key(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="en-US",
        native_mode=True,
    )
    return TranscriptArtifact(
        artifact_key=key,
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        returned_language="en-us",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        segments=[
            {"cue_id": "c0", "start": 0.0, "end": 1.5, "text": "one", "lang": "en"},
            {"cue_id": "c1", "start": 1.5, "end": 3.0, "text": "two", "lang": "en"},
        ],
        duration_sec=3.0,
        created_at=created_at,
    )


def test_generation_budgets_match_fast_and_slow_contracts() -> None:
    fast = GenerationBudget.for_mode("fast")
    slow = GenerationBudget.for_mode("slow")
    assert fast.snapshot()["limits"] == {"search": 3, "transcript": 3, "segmentation": 3}
    assert (fast.max_passes, fast.max_no_growth_passes) == (1, 0)
    assert slow.snapshot()["limits"] == {"search": 12, "transcript": 10, "segmentation": 10}
    assert (slow.max_passes, slow.max_no_growth_passes) == (3, 2)
    for _ in range(3):
        fast.reserve("search")
    with pytest.raises(ProviderBudgetExceededError):
        fast.reserve("search")
    fast.reserve_pass()
    with pytest.raises(ProviderBudgetExceededError):
        fast.reserve_pass()


def test_retry_after_is_parsed_and_bounded() -> None:
    assert bounded_retry_after({"Retry-After": "2.5"}) == 2.5
    assert bounded_retry_after({"retry-after": "999"}) == 30.0
    assert bounded_retry_after({"retry-after": "invalid"}) is None


def test_generation_context_records_billable_headers_and_model_usage() -> None:
    context = GenerationContext("slow", generation_id="job-1")
    context.record_http(
        provider="supadata", operation="search", attempt=1, status_code=200,
        headers={"x-billable-requests": "2"},
    )
    context.record_gemini(
        attempt=1,
        model_used="gemini-primary",
        quality_degraded=False,
        usage={"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
    )
    rows = context.usage()
    assert rows[0]["billable_requests"] == 2
    assert rows[1]["input_tokens"] == 10
    assert rows[1]["output_tokens"] == 4
    assert rows[1]["total_tokens"] == 14
    assert rows[1]["model_used"] == "gemini-primary"


def test_search_cache_key_normalizes_query_filters_and_language_but_not_page() -> None:
    filters = {"preferred_video_duration": "4-20m", "creative_commons_only": True}
    first = search_cache_key(
        query="  CAFÉ  calculus ", filters=filters, language="EN_us", page_token="p1"
    )
    same = search_cache_key(
        query="café calculus", filters=filters, language="en-US", page_token="p1"
    )
    next_page = search_cache_key(
        query="café calculus", filters=filters, language="en-US", page_token="p2"
    )
    assert first == same
    assert first != next_page
    assert normalize_filters(filters) == {
        "duration": "medium",
        "features": ["creative-commons", "subtitles"],
        "sort_by": "relevance",
        "upload_date": "all",
    }
    assert normalize_filters({"creative_commons_only": "false"})["features"] == ["subtitles"]
    assert normalize_filters({"sort_by": "viewCount"})["sort_by"] == "views"


def test_search_cache_uses_shorter_empty_ttl_and_filters_tombstones_twice() -> None:
    cache = MemoryProviderCache()
    positive_key = "positive"
    empty_key = "empty"
    cache.put_search(positive_key, {"videos": [{"id": VIDEO_ID}]}, {})
    cache.put_search(empty_key, {"videos": []}, {})
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    cache.search_rows[positive_key][1]["created_at"] = old
    cache.search_rows[empty_key][1]["created_at"] = old
    assert cache.get_search(positive_key) is not None
    assert cache.get_search(empty_key) is None

    cache.blocked_video_ids.add(VIDEO_ID)
    assert cache.get_search(positive_key).payload["videos"] == []
    cache.put_search("after-block", {"videos": [{"id": VIDEO_ID}]}, {})
    assert cache.search_rows["after-block"][0]["videos"] == []


def test_transcript_validation_rejects_nonfinite_and_nonmonotonic_cues() -> None:
    artifact = _artifact()
    assert validate_transcript_payload(artifact.as_payload()) == artifact
    nonfinite = artifact.as_payload()
    nonfinite["segments"][0]["start"] = float("nan")
    assert validate_transcript_payload(nonfinite) is None
    nonmonotonic = artifact.as_payload()
    nonmonotonic["segments"][1]["start"] = -1
    assert validate_transcript_payload(nonmonotonic) is None


def test_transcript_cache_rejects_tombstoned_video() -> None:
    cache = MemoryProviderCache()
    artifact = _artifact()
    cache.put_transcript(artifact)
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) == artifact
    cache.blocked_video_ids.add(VIDEO_ID)
    assert cache.get_transcript(
        video_id=VIDEO_ID,
        provider="supadata",
        requested_language="en",
        native_mode=True,
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
    ) is None
