"""Versioned DB caches for Supadata search evidence and timed transcript artifacts."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from . import config
from .metadata import normalize_youtube_video_id

logger = logging.getLogger(__name__)

SEARCH_SCHEMA_VERSION = 2
TRANSCRIPT_SCHEMA_VERSION = 4
TRANSCRIPT_PROFILE = f"chunk{max(1, int(config.SUPADATA_CHUNK_SIZE))}-auto"
SEARCH_POSITIVE_TTL_SEC = 6 * 60 * 60
SEARCH_EMPTY_TTL_SEC = 15 * 60
TRANSCRIPT_TTL_SEC = 30 * 24 * 60 * 60
MAX_TRANSCRIPT_DURATION_SEC = 24 * 60 * 60
MAX_CUE_DURATION_SEC = 60 * 60
SUPPORTED_SEARCH_FEATURES = frozenset({
    "hd",
    "creative-commons",
})


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split())


def normalize_language(value: str) -> str:
    return str(value or "").strip().replace("_", "-").casefold()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def normalize_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(filters or {})
    raw_duration = str(
        source.get("duration")
        or source.get("source_duration")
        or source.get("preferred_video_duration")
        or "all"
    ).strip().casefold()
    duration = {
        "short": "short", "under-4m": "short", "under_4m": "short", "<4m": "short",
        "medium": "medium", "4-20m": "medium", "4_20m": "medium",
        "long": "long", "over-20m": "long", "over_20m": "long", ">20m": "long",
    }.get(raw_duration, "all")
    raw_features = {
        str(value).strip().casefold()
        for value in (source.get("features") or [])
        if str(value).strip()
    }
    features = raw_features.intersection(SUPPORTED_SEARCH_FEATURES)
    if _truthy(source.get("creative_commons_only")):
        features.add("creative-commons")
    raw_sort = str(source.get("sortBy") or source.get("sort_by") or "relevance").strip().casefold()
    sort_by = {
        "relevance": "relevance",
        "date": "date",
        "rating": "rating",
        "viewcount": "views",
        "view_count": "views",
        "views": "views",
    }.get(raw_sort, "relevance")
    raw_upload = str(source.get("uploadDate") or source.get("upload_date") or "all").strip().casefold()
    upload_date = raw_upload if raw_upload in {"all", "hour", "today", "week", "month", "year"} else "all"
    return {
        "duration": duration,
        "features": sorted(features),
        "sort_by": sort_by,
        "upload_date": upload_date,
    }


def _video_id_from_result(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    return normalize_youtube_video_id(value.get("id") or value.get("videoId") or value.get("url"))


def _without_blocked_videos(payload: dict[str, Any], blocked_ids: set[str]) -> dict[str, Any]:
    copied = json.loads(_canonical_json(payload))
    videos = copied.get("videos")
    if not isinstance(videos, list) or not blocked_ids:
        return copied
    copied["videos"] = [video for video in videos if _video_id_from_result(video) not in blocked_ids]
    return copied


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def search_cache_key(
    *,
    query: str,
    filters: Mapping[str, Any] | None,
    language: str,
    page_token: str | None,
    provider: str = "supadata",
    schema_version: int = SEARCH_SCHEMA_VERSION,
) -> str:
    payload = {
        "provider": provider.strip().casefold(),
        "schema_version": int(schema_version),
        "query": normalize_query(query),
        "filters": normalize_filters(filters),
        "language": normalize_language(language),
        "page_token": str(page_token or "").strip(),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"supadata-search:v{schema_version}:{digest}"


def transcript_artifact_key(
    *,
    video_id: str,
    provider: str,
    requested_language: str,
    returned_language: str,
    native_mode: bool,
    schema_version: int = TRANSCRIPT_SCHEMA_VERSION,
) -> str:
    payload = {
        "video_id": str(video_id).strip(),
        "provider": str(provider).strip().casefold(),
        "requested_language": normalize_language(requested_language),
        "returned_language": normalize_language(returned_language),
        "native_mode": bool(native_mode),
        "schema_version": int(schema_version),
        "transcript_profile": TRANSCRIPT_PROFILE,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"native-transcript:v{schema_version}:{TRANSCRIPT_PROFILE}:{digest}"


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(created_at: Any, *, now: datetime | None = None) -> float:
    created = _parse_time(created_at)
    if created is None:
        return float("inf")
    return max(0.0, ((now or datetime.now(timezone.utc)) - created).total_seconds())


@dataclass(frozen=True)
class SearchCacheHit:
    payload: dict[str, Any]
    age_sec: float


@dataclass(frozen=True)
class TranscriptArtifact:
    artifact_key: str
    video_id: str
    provider: str
    requested_language: str
    returned_language: str
    native_mode: bool
    schema_version: int
    segments: list[dict[str, Any]]
    duration_sec: float
    created_at: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "artifact_key": self.artifact_key,
            "video_id": self.video_id,
            "provider": self.provider,
            "requested_language": self.requested_language,
            "returned_language": self.returned_language,
            "native_mode": self.native_mode,
            "schema_version": self.schema_version,
            "segments": self.segments,
            "duration_sec": self.duration_sec,
            "created_at": self.created_at,
        }


def validate_transcript_payload(payload: Any) -> TranscriptArtifact | None:
    """Reject the entire artifact on malformed, non-finite, or non-monotonic cues."""
    if not isinstance(payload, Mapping):
        return None
    try:
        schema_version = int(payload.get("schema_version"))
        duration_sec = float(payload.get("duration_sec"))
    except (TypeError, ValueError):
        return None
    if schema_version != TRANSCRIPT_SCHEMA_VERSION:
        return None
    video_id = str(payload.get("video_id") or "").strip()
    provider = str(payload.get("provider") or "").strip().casefold()
    requested = normalize_language(str(payload.get("requested_language") or ""))
    returned = normalize_language(str(payload.get("returned_language") or ""))
    raw_native_mode = payload.get("native_mode")
    if not isinstance(raw_native_mode, bool):
        return None
    native_mode = raw_native_mode
    artifact_key = str(payload.get("artifact_key") or "").strip()
    created_at = str(payload.get("created_at") or "").strip()
    raw_segments = payload.get("segments")
    if (
        not video_id
        or normalize_youtube_video_id(video_id) != video_id
        or provider != "supadata"
        or not requested
        or not returned
        or not artifact_key
        or not created_at
        or _parse_time(created_at) is None
        or not isinstance(raw_segments, list)
        or not raw_segments
        or not math.isfinite(duration_sec)
        or not 0 < duration_sec <= MAX_TRANSCRIPT_DURATION_SEC
    ):
        return None

    segments: list[dict[str, Any]] = []
    previous_start = -1.0
    previous_end = -1.0
    seen_ids: set[str] = set()
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            return None
        try:
            start = float(raw.get("start"))
            end = float(raw.get("end"))
        except (TypeError, ValueError):
            return None
        text = " ".join(str(raw.get("text") or "").split()).strip()
        cue_id = str(raw.get("cue_id") or "").strip()
        if (
            not cue_id
            or cue_id in seen_ids
            or not text
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end - start > MAX_CUE_DURATION_SEC
            or start + 1e-9 < previous_start
            or end + 1e-9 < previous_end
            or end > MAX_TRANSCRIPT_DURATION_SEC
        ):
            return None
        seen_ids.add(cue_id)
        previous_start = start
        previous_end = end
        segments.append(
            {
                "cue_id": cue_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "lang": normalize_language(str(raw.get("lang") or returned)) or returned,
            }
        )
    actual_duration = float(segments[-1]["end"])
    if duration_sec + 1e-3 < actual_duration or duration_sec > max(actual_duration * 1.5, actual_duration + 600):
        return None
    expected_key = transcript_artifact_key(
        video_id=video_id,
        provider=provider,
        requested_language=requested,
        returned_language=returned,
        native_mode=native_mode,
        schema_version=schema_version,
    )
    if artifact_key != expected_key:
        return None
    return TranscriptArtifact(
        artifact_key=artifact_key,
        video_id=video_id,
        provider=provider,
        requested_language=requested,
        returned_language=returned,
        native_mode=native_mode,
        schema_version=schema_version,
        segments=segments,
        duration_sec=duration_sec,
        created_at=created_at,
    )


class ProviderCacheStore(Protocol):
    def get_search(self, cache_key: str) -> SearchCacheHit | None: ...
    def put_search(self, cache_key: str, payload: dict[str, Any], metadata: dict[str, Any]) -> None: ...
    def get_transcript(
        self, *, video_id: str, provider: str, requested_language: str,
        native_mode: bool, schema_version: int,
    ) -> TranscriptArtifact | None: ...
    def put_transcript(self, artifact: TranscriptArtifact) -> None: ...
    def filter_search_payload(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def is_video_tombstoned(self, video_id: str) -> bool: ...


class MemoryProviderCache:
    """Deterministic cache used by focused provider tests."""

    def __init__(self) -> None:
        self.search_rows: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.transcript_rows: dict[str, TranscriptArtifact] = {}
        self.blocked_video_ids: set[str] = set()
        self._lock = threading.Lock()

    def get_search(self, cache_key: str) -> SearchCacheHit | None:
        with self._lock:
            row = self.search_rows.get(cache_key)
        if row is None:
            return None
        payload, metadata = row
        age = _age_seconds(metadata.get("created_at"))
        videos = payload.get("videos") if isinstance(payload, dict) else None
        ttl = SEARCH_POSITIVE_TTL_SEC if isinstance(videos, list) and videos else SEARCH_EMPTY_TTL_SEC
        if age >= ttl:
            return None
        return SearchCacheHit(
            payload=_without_blocked_videos(payload, self.blocked_video_ids), age_sec=age
        )

    def put_search(self, cache_key: str, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
        row_metadata = dict(metadata)
        row_metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self.search_rows[cache_key] = (
                _without_blocked_videos(payload, self.blocked_video_ids), row_metadata
            )

    def get_transcript(
        self, *, video_id: str, provider: str, requested_language: str,
        native_mode: bool, schema_version: int,
    ) -> TranscriptArtifact | None:
        requested = normalize_language(requested_language)
        bare_video_id = normalize_youtube_video_id(video_id) or video_id
        if bare_video_id in self.blocked_video_ids:
            return None
        with self._lock:
            rows = list(self.transcript_rows.values())
        matching = [
            row for row in rows
            if row.video_id == bare_video_id and row.provider == provider
            and row.requested_language == requested and row.native_mode is native_mode
            and row.schema_version == schema_version
        ]
        matching.sort(key=lambda row: row.created_at, reverse=True)
        for row in matching:
            if _age_seconds(row.created_at) >= TRANSCRIPT_TTL_SEC:
                continue
            return validate_transcript_payload(row.as_payload())
        return None

    def put_transcript(self, artifact: TranscriptArtifact) -> None:
        validated = validate_transcript_payload(artifact.as_payload())
        if validated is None:
            raise ValueError("invalid transcript artifact")
        if validated.video_id in self.blocked_video_ids:
            return
        with self._lock:
            self.transcript_rows[artifact.artifact_key] = validated

    def filter_search_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _without_blocked_videos(payload, self.blocked_video_ids)

    def is_video_tombstoned(self, video_id: str) -> bool:
        bare = normalize_youtube_video_id(video_id) or video_id
        return bare in self.blocked_video_ids


class DatabaseProviderCache:
    """Portable DB repository; coordinated migrations create the dedicated tables."""

    def get_search(self, cache_key: str) -> SearchCacheHit | None:
        try:
            from ..db import fetch_one, get_conn
            with get_conn() as conn:
                row = fetch_one(
                    conn,
                    "SELECT response_json, result_count, created_at, expires_at FROM supadata_search_cache WHERE cache_key = ?",
                    (cache_key,),
                )
        except Exception as exc:
            logger.debug("Supadata search cache read unavailable: %s", exc)
            return None
        if not row:
            return None
        try:
            payload = json.loads(str(row.get("response_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("videos"), list):
            return None
        age = _age_seconds(row.get("created_at"))
        ttl = SEARCH_POSITIVE_TTL_SEC if payload["videos"] else SEARCH_EMPTY_TTL_SEC
        if age >= ttl:
            return None
        return SearchCacheHit(
            payload=_without_blocked_videos(payload, self._blocked_ids(payload)), age_sec=age
        )

    def filter_search_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _without_blocked_videos(payload, self._blocked_ids(payload))

    def put_search(self, cache_key: str, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
        created_at = str(metadata.get("created_at") or datetime.now(timezone.utc).isoformat())
        payload = _without_blocked_videos(payload, self._blocked_ids(payload))
        ttl = SEARCH_POSITIVE_TTL_SEC if payload.get("videos") else SEARCH_EMPTY_TTL_SEC
        created = _parse_time(created_at) or datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(created.timestamp() + ttl, tz=timezone.utc).isoformat()
        try:
            from ..db import dumps_json, get_conn, upsert
            with get_conn(transactional=True) as conn:
                upsert(
                    conn,
                    "supadata_search_cache",
                    {
                        "cache_key": cache_key,
                        "provider": "supadata",
                        "schema_version": str(SEARCH_SCHEMA_VERSION),
                        "normalized_query": str(metadata.get("normalized_query") or ""),
                        "filters_json": dumps_json(metadata.get("filters") or {}),
                        "language": str(metadata.get("language") or ""),
                        "page_token": str(metadata.get("page_token") or ""),
                        "response_json": dumps_json(payload),
                        "result_count": len(payload.get("videos") or []),
                        "is_empty": 0 if payload.get("videos") else 1,
                        "created_at": created_at,
                        "expires_at": expires_at,
                    },
                    pk="cache_key",
                )
        except Exception as exc:
            logger.debug("Supadata search cache write unavailable: %s", exc)

    def get_transcript(
        self, *, video_id: str, provider: str, requested_language: str,
        native_mode: bool, schema_version: int,
    ) -> TranscriptArtifact | None:
        requested = normalize_language(requested_language)
        video_id = normalize_youtube_video_id(video_id) or video_id
        if self.is_video_tombstoned(video_id):
            return None
        try:
            from ..db import fetch_one, get_conn
            with get_conn() as conn:
                row = fetch_one(
                    conn,
                    """
                    SELECT cache_key, video_id, provider, requested_language,
                           returned_language, native_mode, schema_version,
                           artifact_json, duration_sec, created_at, expires_at
                    FROM transcript_artifacts
                    WHERE video_id = ? AND provider = ? AND requested_language = ?
                      AND native_mode = ? AND schema_version = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (video_id, provider, requested, 1 if native_mode else 0, str(schema_version)),
                )
                if row:
                    raw_payload = json.loads(str(row.get("artifact_json") or "{}"))
                    payload = raw_payload if isinstance(raw_payload, dict) else None
                else:
                    payload = None
        except Exception as exc:
            logger.debug("Transcript artifact read unavailable: %s", exc)
            return None
        if not isinstance(payload, dict) or _age_seconds(payload.get("created_at") or (row or {}).get("created_at")) >= TRANSCRIPT_TTL_SEC:
            return None
        return validate_transcript_payload(payload)

    def put_transcript(self, artifact: TranscriptArtifact) -> None:
        validated = validate_transcript_payload(artifact.as_payload())
        if validated is None:
            raise ValueError("invalid transcript artifact")
        if self.is_video_tombstoned(validated.video_id):
            return
        payload = validated.as_payload()
        created = _parse_time(validated.created_at) or datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(
            created.timestamp() + TRANSCRIPT_TTL_SEC, tz=timezone.utc
        ).isoformat()
        try:
            from ..db import dumps_json, get_conn, upsert
            with get_conn(transactional=True) as conn:
                upsert(
                    conn,
                    "transcript_artifacts",
                    {
                        "cache_key": validated.artifact_key,
                        "video_id": validated.video_id,
                        "provider": validated.provider,
                        "requested_language": validated.requested_language,
                        "returned_language": validated.returned_language,
                        "native_mode": 1 if validated.native_mode else 0,
                        "schema_version": str(validated.schema_version),
                        "artifact_json": dumps_json(payload),
                        "cue_count": len(validated.segments),
                        "duration_sec": validated.duration_sec,
                        "created_at": validated.created_at,
                        "expires_at": expires_at,
                    },
                    pk="cache_key",
                )
        except Exception as exc:
            logger.debug("Transcript artifact write unavailable: %s", exc)

    def _blocked_ids(self, payload: Mapping[str, Any]) -> set[str]:
        videos = payload.get("videos")
        ids = {
            video_id for video_id in (_video_id_from_result(item) for item in videos or [])
            if video_id
        }
        if not ids:
            return set()
        try:
            from ..db import fetch_all, get_conn
            placeholders = ", ".join("?" for _ in ids)
            with get_conn() as conn:
                rows = fetch_all(
                    conn,
                    f"SELECT video_id FROM blocked_video_tombstones WHERE video_id IN ({placeholders})",
                    tuple(sorted(ids)),
                )
            return {str(row.get("video_id") or "") for row in rows}
        except Exception as exc:
            logger.debug("Blocked-video lookup unavailable: %s", exc)
            return set()

    def is_video_tombstoned(self, video_id: str) -> bool:
        bare = normalize_youtube_video_id(video_id)
        if bare is None:
            return False
        try:
            from ..db import fetch_one, get_conn
            with get_conn() as conn:
                return bool(fetch_one(
                    conn,
                    "SELECT video_id FROM blocked_video_tombstones WHERE video_id = ?",
                    (bare,),
                ))
        except Exception as exc:
            logger.debug("Blocked-video lookup unavailable: %s", exc)
            return False


DEFAULT_PROVIDER_CACHE: ProviderCacheStore = DatabaseProviderCache()
