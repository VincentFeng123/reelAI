"""End-to-end regression for captionless YouTube topic ingestion.

The provider boundaries are mocked, while transcript normalization, Gemini's
quality gates, the final topic gate, and reel persistence all run normally.
"""

from __future__ import annotations

from typing import Any

from backend.app import db as db_module
from backend.app.clip_engine.clipper import supadata_client
from backend.app.clip_engine.provider_cache import MemoryProviderCache
from backend.app.clip_engine.provider_runtime import GenerationContext
from backend.app.config import get_settings
from backend.app.ingestion import pipeline as pipeline_module
from backend.app.ingestion.pipeline import IngestionPipeline, _PlatformRateLimiter
from backend.app.services.search_query_plan import PlannedSearchQuery, SearchQueryPlan
from backend.pipeline import gemini_segment


VIDEO_ID = "PyIntro0001"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
TOPIC = "intro to Python"


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


def test_captionless_candidate_becomes_persisted_timestamped_embed(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    db_module._db_ready = False

    with db_module.get_conn(transactional=True) as conn:
        db_module.insert(
            conn,
            "materials",
            {
                "id": "mat-captionless",
                "subject_tag": TOPIC,
                "raw_text": TOPIC,
                "source_type": "topic",
                "source_path": None,
                "created_at": db_module.now_iso(),
            },
        )
        db_module.insert(
            conn,
            "concepts",
            {
                "id": "con-captionless",
                "material_id": "mat-captionless",
                "title": TOPIC,
                "keywords_json": "[]",
                "summary": "",
                "embedding_json": None,
                "created_at": db_module.now_iso(),
            },
        )

    query_plan = SearchQueryPlan(
        literal_query=TOPIC,
        canonical_query=TOPIC,
        trusted_signature=["Python"],
        provenance={"intro to python": ["literal"]},
        queries=[
            PlannedSearchQuery(
                text=TOPIC,
                family="python",
                provenance="literal",
                trust="literal",
            )
        ],
    )

    def fake_discover(topic: str, **_kwargs: Any) -> dict[str, Any]:
        assert topic == TOPIC
        return {
            "corrected": TOPIC,
            "videos": [
                {
                    "id": VIDEO_ID,
                    "url": VIDEO_URL,
                    "title": "Python for Absolute Beginners",
                    "channel": "Clear Code Lessons",
                    "duration": 90.0,
                    "thumbnail": "",
                    "view_count": 500_000,
                    "upload_date": "2026-01-01",
                    "matched_queries": [TOPIC],
                    "matched_families": ["python"],
                    "matched_query_provenance": {TOPIC: ["literal"]},
                }
            ],
            "query_plan": query_plan,
            "credits_used": 0,
            "warning": None,
        }

    monkeypatch.setattr(pipeline_module.clip_engine_search, "discover", fake_discover)

    provider_calls: list[tuple[str, dict[str, Any] | None]] = []

    class FakeSupadataClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def get(
            self,
            url: str,
            params: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> _Response:
            provider_calls.append((url, params))
            if url.endswith("/transcript"):
                return _Response(202, {"jobId": "captionless-job"})
            assert url.endswith("/transcript/captionless-job")
            return _Response(
                200,
                {
                    "status": "completed",
                    "result": {
                        "lang": "en",
                        "content": [
                            {
                                "id": "generated-0",
                                "offset": 5_000,
                                "duration": 10_000,
                                "text": "Python variables store values such as numbers and strings.",
                            },
                            {
                                "id": "generated-1",
                                "offset": 15_000,
                                "duration": 15_000,
                                "text": "In Python, assign a variable with equals and print the value.",
                            },
                            {
                                "id": "generated-2",
                                "offset": 30_000,
                                "duration": 15_000,
                                "text": "This makes a first Python program easy to understand.",
                            },
                        ],
                    },
                },
            )

    async def no_poll_delay(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(supadata_client.httpx, "AsyncClient", FakeSupadataClient)
    monkeypatch.setattr(supadata_client, "sleep_with_probe", no_poll_delay)
    monkeypatch.setattr(supadata_client.config, "SUPADATA_API_KEY", "sd_test")

    gemini_calls: list[dict[str, Any]] = []

    def fake_model_call(
        system: str,
        user: str,
        schema,
        **kwargs: Any,
    ):
        gemini_calls.append({"system": system, "user": user, **kwargs})
        assert schema is gemini_segment._BoundaryPlan
        plan = schema.model_validate(
            {
                "topics": [
                        {
                            "candidate_id": "python-variables",
                            "title": "Python variables and assignment",
                            "learning_objective": (
                                "Understand how Python variables store and print values."
                            ),
                        "start_line": 0,
                        "end_line": 2,
                        "start_quote": "Python variables store values",
                        "end_quote": "first Python program easy to understand",
                        "reason": "A complete beginner explanation.",
                        "facet": "variables",
                            "informativeness": 0.95,
                            "topic_relevance": 0.99,
                                "educational_importance": 0.97,
                                    "directly_teaches_topic": True,
                                    "substantive": True,
                                    "factually_grounded": True,
                                    "topic_evidence_quote": (
                                    "Python variables store values such as numbers and strings"
                                ),
                                "self_contained": True,
                            "is_standalone": True,
                            "prerequisite_candidate_ids": [],
                        "difficulty": 0.1,
                        "uncertainty": "low",
                        "uncertainty_reasons": [],
                    }
                ]
            }
        )
        return plan, {
            "model": kwargs["model"],
            "operation": kwargs["operation"],
            "prompt_version": kwargs["prompt_version"],
            "prompt_token_count": 100,
            "candidates_token_count": 20,
            "total_token_count": 120,
            "retries": 0,
        }

    monkeypatch.setattr(gemini_segment, "_call_model", fake_model_call)
    monkeypatch.setattr(gemini_segment, "_flash_disabled_reason", None)
    monkeypatch.setattr(gemini_segment.config, "SEGMENT_ROUTING_MODE", "hybrid")
    monkeypatch.setattr(gemini_segment.config, "SEGMENT_HYBRID_PERCENT", 100.0)

    cache = MemoryProviderCache()
    context = GenerationContext("fast", cache_store=cache)
    pipeline = IngestionPipeline(
        youtube_service=object(),
        embedding_service=object(),
        settings=get_settings(),
        rate_limiter=_PlatformRateLimiter(overrides={"yt": (1000, 60.0)}),
        serverless_mode=False,
    )

    try:
        reels, resolved_video_ids = pipeline.ingest_topic(
            topic=TOPIC,
            material_id="mat-captionless",
            concept_id="con-captionless",
            generation_id="gen-captionless",
            max_videos=1,
            max_reels=1,
            generation_context=context,
        )

        assert resolved_video_ids == [VIDEO_ID]
        assert len(reels) == 1
        reel = reels[0]
        assert reel.video_url == (
            f"https://www.youtube.com/embed/{VIDEO_ID}"
            "?start=4&end=45&modestbranding=1&rel=0&playsinline=1"
        )
        assert (reel.t_start, reel.t_end) == (4.7, 45.0)
        captions = [cue.model_dump() for cue in reel.captions]
        for cue in captions:
            cue["start"] = round(float(cue["start"]), 3)
            cue["end"] = round(float(cue["end"]), 3)
        assert captions == [
            {"start": 0.3, "end": 10.3, "text": "Python variables store values such as numbers and strings."},
            {"start": 10.3, "end": 25.3, "text": "In Python, assign a variable with equals and print the value."},
            {"start": 25.3, "end": 40.3, "text": "This makes a first Python program easy to understand."},
        ]

        assert provider_calls[0][0].endswith("/transcript")
        assert provider_calls[0][1] == {
            "url": VIDEO_URL,
            "text": "false",
            "mode": "auto",
            "lang": "en",
            "chunkSize": "50",
        }
        assert provider_calls[1][0].endswith("/transcript/captionless-job")
        assert gemini_calls[0]["prompt_version"] == gemini_segment.PRODUCTION_FLASH_PROFILE
        assert "timestamped transcripts" in gemini_calls[0]["system"]
        assert "[0] 00:05 Python variables store values" in gemini_calls[0]["user"]

        artifacts = list(cache.transcript_rows.values())
        assert len(artifacts) == 1
        assert artifacts[0].native_mode is False
        assert artifacts[0].segments[0]["cue_id"] == "generated-0"

        with db_module.get_conn() as conn:
            persisted = db_module.fetch_one(
                conn,
                "SELECT video_url, t_start, t_end, transcript_snippet "
                "FROM reels WHERE generation_id = ?",
                ("gen-captionless",),
            )
        assert persisted is not None
        assert persisted["video_url"] == reel.video_url
        assert (persisted["t_start"], persisted["t_end"]) == (4.7, 45.0)
        assert "Python variables" in persisted["transcript_snippet"]
    finally:
        db_module._db_ready = False
        get_settings.cache_clear()
