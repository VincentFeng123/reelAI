import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from .config import get_settings
from .db import dumps_json, fetch_all, get_conn, init_db, now_iso, upsert
from .models import (
    ChatRequest,
    ChatResponse,
    CommunityReelOut,
    CommunitySetCreateRequest,
    CommunitySetOut,
    CommunitySetUpdateRequest,
    CommunitySetsResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedResponse,
    MaterialResponse,
    ReelsCanGenerateAnyRequest,
    ReelsCanGenerateAnyResponse,
    ReelsCanGenerateResponse,
    ReelsGenerateRequest,
    ReelsGenerateResponse,
)
from .services.embeddings import EmbeddingService
from .services.material_intelligence import MaterialIntelligenceService
from .services.parsers import ParseError, extract_text_from_file
from .services.reels import ReelService
from .services.storage import get_storage
from .services.text_utils import chunk_text, normalize_whitespace
from .services.youtube import YouTubeApiRequestError, YouTubeService, parse_iso8601_duration

settings = get_settings()
app = FastAPI(title="StudyReels API", version="0.1.0")

def _build_allowed_origins() -> list[str]:
    local_defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
    env_origins = [
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
        if origin.strip()
    ]
    candidates = [settings.frontend_origin, *local_defaults, *env_origins]
    normalized: list[str] = []
    seen: set[str] = set()
    for origin in candidates:
        clean = str(origin or "").strip().rstrip("/")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


allowed_origins = _build_allowed_origins()
allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", r"https?://.*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = get_storage()
embedding_service = EmbeddingService()
material_intelligence_service = MaterialIntelligenceService()
youtube_service = YouTubeService()
reel_service = ReelService(embedding_service=embedding_service, youtube_service=youtube_service)
SERVERLESS_MODE = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE"))

VALID_VIDEO_POOL_MODES = {"short-first", "balanced", "long-form"}
VALID_VIDEO_DURATION_PREFS = {"any", "short", "medium", "long"}
DEFAULT_TARGET_CLIP_DURATION_SEC = 55
MIN_TARGET_CLIP_DURATION_SEC = 15
MAX_TARGET_CLIP_DURATION_SEC = 180
MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC = 15
MAX_COMMUNITY_REEL_DURATION_SEC = 8 * 60 * 60
COMMUNITY_REEL_DURATION_TIMEOUT_SEC = 6.0


def _normalize_duration_seconds(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    if parsed > MAX_COMMUNITY_REEL_DURATION_SEC:
        return None
    return round(parsed, 1)


def _extract_duration_from_html(html: str) -> float | None:
    if not html:
        return None

    meta_duration_patterns = [
        r'<meta[^>]*property=["\']video:duration["\'][^>]*content=["\'](\d+(?:\.\d+)?)["\'][^>]*>',
        r'<meta[^>]*content=["\'](\d+(?:\.\d+)?)["\'][^>]*property=["\']video:duration["\'][^>]*>',
        r'<meta[^>]*name=["\']video:duration["\'][^>]*content=["\'](\d+(?:\.\d+)?)["\'][^>]*>',
        r'<meta[^>]*content=["\'](\d+(?:\.\d+)?)["\'][^>]*name=["\']video:duration["\'][^>]*>',
    ]
    for pattern in meta_duration_patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if not match:
            continue
        duration = _normalize_duration_seconds(match.group(1))
        if duration is not None:
            return duration

    length_seconds_match = re.search(r'"lengthSeconds"\s*:\s*"(\d{1,7})"', html)
    if length_seconds_match:
        duration = _normalize_duration_seconds(length_seconds_match.group(1))
        if duration is not None:
            return duration

    json_ld_duration = re.search(r'"duration"\s*:\s*"((?:P|PT)[^"]+)"', html, flags=re.IGNORECASE)
    if json_ld_duration:
        iso_duration = parse_iso8601_duration(json_ld_duration.group(1))
        duration = _normalize_duration_seconds(iso_duration)
        if duration is not None:
            return duration

    generic_numeric_duration = re.search(r'"duration"\s*:\s*([0-9]{1,7}(?:\.[0-9]+)?)', html)
    if generic_numeric_duration:
        duration = _normalize_duration_seconds(generic_numeric_duration.group(1))
        if duration is not None:
            return duration

    return None


def _fetch_duration_from_source_page(source_url: str) -> float | None:
    try:
        response = requests.get(
            source_url,
            timeout=COMMUNITY_REEL_DURATION_TIMEOUT_SEC,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        response.raise_for_status()
    except requests.RequestException:
        return None
    html = response.text[:900_000]
    return _extract_duration_from_html(html)


def _resolve_community_reel_duration_sec(source_url: str) -> float | None:
    parsed = urlparse(source_url)
    host = (parsed.hostname or "").lower()

    if "youtube.com" in host or "youtu.be" in host:
        video_id = youtube_service._extract_video_id_from_url(source_url)  # noqa: SLF001
        if video_id:
            details = youtube_service._video_details([video_id])  # noqa: SLF001
            duration = _normalize_duration_seconds((details.get(video_id) or {}).get("duration_sec"))
            if duration is not None:
                return duration

    return _fetch_duration_from_source_page(source_url)


def _normalize_community_tags(raw_tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_tags:
        value = str(raw).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= 6:
            break
    return normalized


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_clip_seconds(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return round(parsed, 3)


def _normalize_min_relevance(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(-1.0, min(1.2, parsed))


def _normalize_video_pool_mode(value: str | None) -> Literal["short-first", "balanced", "long-form"]:
    if value in VALID_VIDEO_POOL_MODES:
        return value
    return "short-first"


def _normalize_preferred_video_duration(value: str | None) -> Literal["any", "short", "medium", "long"]:
    if value in VALID_VIDEO_DURATION_PREFS:
        return value
    return "any"


def _normalize_target_clip_duration_sec(value: int | float | None) -> int:
    if value is None:
        return DEFAULT_TARGET_CLIP_DURATION_SEC
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TARGET_CLIP_DURATION_SEC
    return max(MIN_TARGET_CLIP_DURATION_SEC, min(MAX_TARGET_CLIP_DURATION_SEC, parsed))


def _resolve_target_clip_duration_bounds(
    target_clip_duration_sec: int | float | None,
    target_clip_duration_min_sec: int | float | None,
    target_clip_duration_max_sec: int | float | None,
) -> tuple[int, int, int]:
    safe_target = _normalize_target_clip_duration_sec(target_clip_duration_sec)
    default_min = max(10, int(round(safe_target * 0.35)))
    default_max = max(default_min + MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC, safe_target)

    safe_min = default_min if target_clip_duration_min_sec is None else _normalize_target_clip_duration_sec(
        target_clip_duration_min_sec
    )
    safe_max = default_max if target_clip_duration_max_sec is None else _normalize_target_clip_duration_sec(
        target_clip_duration_max_sec
    )
    if safe_min > safe_max:
        safe_min, safe_max = safe_max, safe_min
    if safe_max - safe_min < MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC:
        expanded_max = min(MAX_TARGET_CLIP_DURATION_SEC, safe_min + MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC)
        if expanded_max - safe_min >= MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC:
            safe_max = expanded_max
        else:
            safe_min = max(MIN_TARGET_CLIP_DURATION_SEC, safe_max - MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC)
    safe_target = max(safe_min, min(safe_max, safe_target))
    return safe_target, safe_min, safe_max


def _video_duration_bucket(duration_sec: object) -> Literal["short", "medium", "long"] | None:
    try:
        parsed = int(duration_sec)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    if parsed < 4 * 60:
        return "short"
    if parsed <= 20 * 60:
        return "medium"
    return "long"


def _filter_reels_by_min_relevance(reels: list[dict], min_relevance: float | None) -> list[dict]:
    if min_relevance is None:
        return reels
    filtered: list[dict] = []
    for reel in reels:
        relevance = reel.get("relevance_score")
        if isinstance(relevance, (int, float)) and float(relevance) < min_relevance:
            continue
        filtered.append(reel)
    return filtered


def _filter_reels_by_video_preferences(
    reels: list[dict],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None = None,
    target_clip_duration_max_sec: int | None = None,
) -> list[dict]:
    safe_duration_pref = _normalize_preferred_video_duration(preferred_video_duration)
    _, clip_min, clip_max = _resolve_target_clip_duration_bounds(
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
    )

    filtered: list[dict] = []
    for reel in reels:
        if safe_duration_pref != "any":
            bucket = _video_duration_bucket(reel.get("video_duration_sec"))
            if bucket != safe_duration_pref:
                continue

        clip_duration = reel.get("clip_duration_sec")
        if not isinstance(clip_duration, (int, float)):
            try:
                clip_duration = float(reel.get("t_end") or 0) - float(reel.get("t_start") or 0)
            except (TypeError, ValueError):
                clip_duration = 0.0
        clip_duration_value = float(clip_duration or 0.0)
        if clip_duration_value > 0 and (clip_duration_value < clip_min or clip_duration_value > clip_max):
            continue

        normalized = dict(reel)
        if clip_duration_value > 0:
            normalized["clip_duration_sec"] = round(clip_duration_value, 2)
        filtered.append(normalized)
    return filtered


def _serialize_community_set(row: dict) -> CommunitySetOut:
    tags_raw = row.get("tags_json")
    reels_raw = row.get("reels_json")
    try:
        decoded_tags = json.loads(tags_raw) if isinstance(tags_raw, str) else []
    except json.JSONDecodeError:
        decoded_tags = []
    try:
        decoded_reels = json.loads(reels_raw) if isinstance(reels_raw, str) else []
    except json.JSONDecodeError:
        decoded_reels = []

    tags = _normalize_community_tags(decoded_tags if isinstance(decoded_tags, list) else [])
    reels: list[CommunityReelOut] = []
    if isinstance(decoded_reels, list):
        for index, entry in enumerate(decoded_reels):
            if not isinstance(entry, dict):
                continue
            platform = str(entry.get("platform", "")).strip().lower()
            if platform not in {"youtube", "instagram", "tiktok"}:
                continue
            source_url = str(entry.get("source_url", "")).strip()
            embed_url = str(entry.get("embed_url", "")).strip()
            if not source_url or not embed_url:
                continue
            reel_id = str(entry.get("id") or f"{row.get('id', 'community-set')}-reel-{index}")
            t_start_sec = _normalize_clip_seconds(entry.get("t_start_sec"))
            t_end_sec = _normalize_clip_seconds(entry.get("t_end_sec"))
            if t_end_sec is not None and t_start_sec is not None and t_end_sec <= t_start_sec:
                t_end_sec = None
            reels.append(
                CommunityReelOut(
                    id=reel_id,
                    platform=platform,
                    source_url=source_url,
                    embed_url=embed_url,
                    t_start_sec=t_start_sec,
                    t_end_sec=t_end_sec,
                )
            )

    reel_count = max(len(reels), _to_int(row.get("reel_count"), len(reels)))
    likes = max(0, _to_int(row.get("likes"), 0))
    learners = max(0, _to_int(row.get("learners"), 1))
    updated_label = str(row.get("updated_label") or "Updated just now").strip() or "Updated just now"
    curator = str(row.get("curator") or "Community member").strip() or "Community member"
    thumbnail_url = str(row.get("thumbnail_url") or "").strip()
    if not thumbnail_url:
        thumbnail_url = "/images/community/ai-systems.svg"

    return CommunitySetOut(
        id=str(row.get("id") or ""),
        title=str(row.get("title") or "").strip(),
        description=str(row.get("description") or "").strip(),
        tags=tags,
        reels=reels,
        reel_count=reel_count,
        curator=curator,
        likes=likes,
        learners=learners,
        updated_label=updated_label,
        thumbnail_url=thumbnail_url,
        featured=bool(_to_int(row.get("featured"), 0)),
    )


@app.on_event("startup")
def on_startup() -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/material", response_model=MaterialResponse)
async def create_material(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    subject_tag: str | None = Form(default=None),
):
    if not file and not text and not subject_tag:
        raise HTTPException(status_code=400, detail="Provide at least one of: topic, text, or file")

    source_parts: list[str] = []
    source_type = "mixed"
    source_path = None

    if file:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            file_text = extract_text_from_file(file.filename or "upload.txt", content)
        except ParseError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if file_text.strip():
            source_parts.append(file_text)
        source_path = storage.save_bytes(content, file.filename or "material")
        source_type = "file"

    if text and text.strip():
        source_parts.append(text)
        if source_type != "file":
            source_type = "text"

    if subject_tag and subject_tag.strip():
        source_parts.append(f"Topic: {subject_tag.strip()}")
        if source_type not in {"file", "text"}:
            source_type = "topic"

    raw_text = "\n\n".join(source_parts)
    raw_text = normalize_whitespace(raw_text)
    if len(raw_text) < 3:
        raise HTTPException(status_code=400, detail="Input is too short")

    chunks = chunk_text(raw_text)
    if SERVERLESS_MODE and len(chunks) > 64:
        chunks = chunks[:64]

    material_id = str(uuid.uuid4())
    created_at = now_iso()

    with get_conn() as conn:
        concept_limit = 6 if SERVERLESS_MODE else 12
        concepts, objectives = material_intelligence_service.extract_concepts_and_objectives(
            conn,
            raw_text,
            subject_tag=subject_tag,
            max_concepts=concept_limit,
        )
        if objectives and len(concepts) < concept_limit and not any(c["title"].lower() == "learning objectives" for c in concepts):
            concepts.append(
                {
                    "id": str(uuid.uuid4()),
                    "title": "Learning Objectives",
                    "keywords": [o[:40] for o in objectives][:5],
                    "summary": objectives[0][:240],
                }
            )

        upsert(
            conn,
            "materials",
            {
                "id": material_id,
                "subject_tag": subject_tag,
                "raw_text": raw_text,
                "source_type": source_type,
                "source_path": source_path,
                "created_at": created_at,
            },
        )

        for concept in concepts:
            concept_text = (
                f"{concept['title']}. Keywords: {' '.join(concept['keywords'])}. Summary: {concept['summary']}"
            )
            emb = embedding_service.embed_texts(conn, [concept_text])[0]
            upsert(
                conn,
                "concepts",
                {
                    "id": concept["id"],
                    "material_id": material_id,
                    "title": concept["title"],
                    "keywords_json": dumps_json(concept["keywords"]),
                    "summary": concept["summary"],
                    "embedding_json": dumps_json(emb.tolist()),
                    "created_at": created_at,
                },
            )

        if chunks:
            chunk_embeddings = embedding_service.embed_texts(conn, chunks)
            for i, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
                upsert(
                    conn,
                    "material_chunks",
                    {
                        "id": str(uuid.uuid4()),
                        "material_id": material_id,
                        "chunk_index": i,
                        "text": chunk,
                        "embedding_json": dumps_json(emb.tolist()),
                        "created_at": created_at,
                    },
                )

    return {
        "material_id": material_id,
        "extracted_concepts": concepts,
    }


@app.post("/api/reels/generate", response_model=ReelsGenerateResponse)
def generate_reels(payload: ReelsGenerateRequest):
    min_relevance = _normalize_min_relevance(payload.min_relevance)
    safe_video_pool_mode = _normalize_video_pool_mode(payload.video_pool_mode)
    safe_video_duration_pref = _normalize_preferred_video_duration(payload.preferred_video_duration)
    safe_target_clip_duration_sec, safe_target_clip_min_sec, safe_target_clip_max_sec = _resolve_target_clip_duration_bounds(
        target_clip_duration_sec=payload.target_clip_duration_sec,
        target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
        target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
    )
    requested_num_reels = max(1, int(payload.num_reels))
    if SERVERLESS_MODE:
        # Keep hosted/serverless requests within function time limits.
        requested_num_reels = min(requested_num_reels, 6)

    reels: list[dict] = []
    filtered: list[dict] = []
    with get_conn() as conn:
        material = fetch_all(conn, "SELECT id FROM materials WHERE id = ?", (payload.material_id,))
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")

        attempts = 0
        max_attempts = 1 if SERVERLESS_MODE else 4
        effective_fast_mode = True if SERVERLESS_MODE else payload.generation_mode == "fast"
        while attempts < max_attempts and len(filtered) < requested_num_reels:
            need = max(1, requested_num_reels - len(filtered))
            if SERVERLESS_MODE:
                target_batch = min(8, max(need, 3))
            else:
                target_batch = min(30, max(need + 2, requested_num_reels if attempts == 0 else need))
            try:
                batch = reel_service.generate_reels(
                    conn,
                    material_id=payload.material_id,
                    concept_id=payload.concept_id,
                    num_reels=target_batch,
                    creative_commons_only=payload.creative_commons_only,
                    fast_mode=effective_fast_mode,
                    video_pool_mode=safe_video_pool_mode,
                    preferred_video_duration=safe_video_duration_pref,
                    target_clip_duration_sec=safe_target_clip_duration_sec,
                    target_clip_duration_min_sec=safe_target_clip_min_sec,
                    target_clip_duration_max_sec=safe_target_clip_max_sec,
                )
            except YouTubeApiRequestError as e:
                raise HTTPException(status_code=502, detail=str(e)) from e
            attempts += 1
            if not batch:
                break
            reels.extend(batch)
            filtered = _filter_reels_by_min_relevance(reels, min_relevance)
            filtered = _filter_reels_by_video_preferences(
                filtered,
                preferred_video_duration=safe_video_duration_pref,
                target_clip_duration_sec=safe_target_clip_duration_sec,
                target_clip_duration_min_sec=safe_target_clip_min_sec,
                target_clip_duration_max_sec=safe_target_clip_max_sec,
            )
    return {"reels": filtered[:requested_num_reels]}


@dataclass
class ProbeResult:
    can_generate: bool = False
    blocked_by_settings: bool = False
    total_probed: int = 0
    passed_relevance: int = 0
    passed_duration_pref: int = 0
    passed_clip_range: int = 0
    passed_all: int = 0


def _filter_reels_by_relevance_only(reels: list[dict], min_relevance: float | None) -> list[dict]:
    if min_relevance is None:
        return list(reels)
    return [
        r for r in reels
        if not (isinstance(r.get("relevance_score"), (int, float)) and float(r["relevance_score"]) < min_relevance)
    ]


def _filter_reels_by_duration_pref_only(
    reels: list[dict],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
) -> list[dict]:
    safe_pref = _normalize_preferred_video_duration(preferred_video_duration)
    if safe_pref == "any":
        return list(reels)
    return [
        r for r in reels
        if _video_duration_bucket(r.get("video_duration_sec")) == safe_pref
    ]


def _filter_reels_by_clip_range_only(
    reels: list[dict],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
) -> list[dict]:
    _, clip_min, clip_max = _resolve_target_clip_duration_bounds(
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
    )
    filtered: list[dict] = []
    for reel in reels:
        clip_duration = reel.get("clip_duration_sec")
        if not isinstance(clip_duration, (int, float)):
            try:
                clip_duration = float(reel.get("t_end") or 0) - float(reel.get("t_start") or 0)
            except (TypeError, ValueError):
                clip_duration = 0.0
        val = float(clip_duration or 0.0)
        if val > 0 and (val < clip_min or val > clip_max):
            continue
        filtered.append(reel)
    return filtered


def _determine_primary_bottleneck(
    total: int,
    passed_relevance: int,
    passed_duration_pref: int,
    passed_clip_range: int,
) -> str:
    if total == 0:
        return "no_source"
    dropped_by_relevance = total - passed_relevance
    dropped_by_duration = total - passed_duration_pref
    dropped_by_clip = total - passed_clip_range
    if dropped_by_relevance == 0 and dropped_by_duration == 0 and dropped_by_clip == 0:
        return ""
    worst = max(dropped_by_relevance, dropped_by_duration, dropped_by_clip)
    if worst == 0:
        return ""
    if dropped_by_relevance == worst:
        return "relevance"
    if dropped_by_duration == worst:
        return "video_duration"
    return "clip_range"


def _probe_material_viability(
    conn,
    *,
    material_id: str,
    concept_id: str | None,
    creative_commons_only: bool,
    fast_mode: bool,
    min_relevance: float | None,
    video_pool_mode: Literal["short-first", "balanced", "long-form"],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
) -> ProbeResult:
    probe: list[dict] = []
    batch = reel_service.generate_reels(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        num_reels=4,
        creative_commons_only=creative_commons_only,
        fast_mode=fast_mode,
        video_pool_mode=video_pool_mode,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        dry_run=True,
    )
    if batch:
        probe.extend(batch)

    total_probed = len(probe)

    # Apply each filter independently to measure per-filter pass rates
    after_relevance = _filter_reels_by_relevance_only(probe, min_relevance)
    after_duration = _filter_reels_by_duration_pref_only(probe, preferred_video_duration)
    after_clip = _filter_reels_by_clip_range_only(
        probe,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
    )

    # Apply all filters combined
    all_filtered = _filter_reels_by_min_relevance(probe, min_relevance)
    all_filtered = _filter_reels_by_video_preferences(
        all_filtered,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
    )

    passed_relevance = len(after_relevance)
    passed_duration = len(after_duration)
    passed_clip = len(after_clip)
    passed_all = len(all_filtered)

    if passed_all > 0:
        return ProbeResult(
            can_generate=True,
            blocked_by_settings=False,
            total_probed=total_probed,
            passed_relevance=passed_relevance,
            passed_duration_pref=passed_duration,
            passed_clip_range=passed_clip,
            passed_all=passed_all,
        )

    # Relaxed probe to determine if settings are the blocker
    relaxed_probe = reel_service.generate_reels(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        num_reels=1,
        creative_commons_only=creative_commons_only,
        fast_mode=fast_mode,
        video_pool_mode="balanced",
        preferred_video_duration="any",
        target_clip_duration_sec=DEFAULT_TARGET_CLIP_DURATION_SEC,
        target_clip_duration_min_sec=MIN_TARGET_CLIP_DURATION_SEC,
        target_clip_duration_max_sec=MAX_TARGET_CLIP_DURATION_SEC,
        dry_run=True,
    )

    blocked = bool(relaxed_probe)
    return ProbeResult(
        can_generate=False,
        blocked_by_settings=blocked,
        total_probed=total_probed,
        passed_relevance=passed_relevance,
        passed_duration_pref=passed_duration,
        passed_clip_range=passed_clip,
        passed_all=0,
    )


@app.post("/api/reels/can-generate", response_model=ReelsCanGenerateResponse)
def can_generate_reels(payload: ReelsGenerateRequest):
    min_relevance = _normalize_min_relevance(payload.min_relevance)
    safe_video_pool_mode = _normalize_video_pool_mode(payload.video_pool_mode)
    safe_video_duration_pref = _normalize_preferred_video_duration(payload.preferred_video_duration)
    safe_target_clip_duration_sec, safe_target_clip_min_sec, safe_target_clip_max_sec = _resolve_target_clip_duration_bounds(
        target_clip_duration_sec=payload.target_clip_duration_sec,
        target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
        target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
    )

    with get_conn() as conn:
        material = fetch_all(conn, "SELECT id FROM materials WHERE id = ?", (payload.material_id,))
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")

        try:
            result = _probe_material_viability(
                conn,
                material_id=payload.material_id,
                concept_id=payload.concept_id,
                creative_commons_only=payload.creative_commons_only,
                fast_mode=payload.generation_mode == "fast",
                min_relevance=min_relevance,
                video_pool_mode=safe_video_pool_mode,
                preferred_video_duration=safe_video_duration_pref,
                target_clip_duration_sec=safe_target_clip_duration_sec,
                target_clip_duration_min_sec=safe_target_clip_min_sec,
                target_clip_duration_max_sec=safe_target_clip_max_sec,
            )
        except YouTubeApiRequestError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    success_rate = result.passed_all / result.total_probed if result.total_probed > 0 else 0.0
    bottleneck = _determine_primary_bottleneck(
        result.total_probed, result.passed_relevance, result.passed_duration_pref, result.passed_clip_range,
    )
    base = {
        "estimated_success_rate": round(success_rate, 4),
        "total_probed": result.total_probed,
        "passed_all_filters": result.passed_all,
        "primary_bottleneck": bottleneck,
    }

    if result.can_generate:
        return {
            **base,
            "can_generate": True,
            "blocked_by_settings": False,
            "message": "Current settings can generate reels for this material.",
        }

    if result.blocked_by_settings:
        return {
            **base,
            "can_generate": False,
            "blocked_by_settings": True,
            "message": "This configuration is too strict for the current material.",
        }

    return {
        **base,
        "can_generate": False,
        "blocked_by_settings": False,
        "message": "No matching source videos were found for this material right now.",
    }


@app.post("/api/reels/can-generate-any", response_model=ReelsCanGenerateAnyResponse)
def can_generate_reels_any(payload: ReelsCanGenerateAnyRequest):
    min_relevance = _normalize_min_relevance(payload.min_relevance)
    safe_video_pool_mode = _normalize_video_pool_mode(payload.video_pool_mode)
    safe_video_duration_pref = _normalize_preferred_video_duration(payload.preferred_video_duration)
    safe_target_clip_duration_sec, safe_target_clip_min_sec, safe_target_clip_max_sec = _resolve_target_clip_duration_bounds(
        target_clip_duration_sec=payload.target_clip_duration_sec,
        target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
        target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
    )

    with get_conn() as conn:
        requested_ids: list[str] = []
        seen_requested: set[str] = set()
        for raw in payload.material_ids:
            material_id = str(raw or "").strip()
            if not material_id or material_id in seen_requested:
                continue
            seen_requested.add(material_id)
            requested_ids.append(material_id)

        if requested_ids:
            placeholders = ",".join(["?"] * len(requested_ids))
            rows = fetch_all(
                conn,
                f"SELECT id FROM materials WHERE id IN ({placeholders})",
                tuple(requested_ids),
            )
            existing = {str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()}
            material_ids = [material_id for material_id in requested_ids if material_id in existing]
        else:
            rows = fetch_all(conn, "SELECT id FROM materials ORDER BY created_at DESC LIMIT 5", tuple())
            material_ids = [str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()]

        # Cap to avoid timeout from too many external API calls per probe
        material_ids = material_ids[:5]

        if not material_ids:
            return {
                "can_generate_any": False,
                "topics_checked": 0,
                "topics_can_generate": 0,
                "blocked_by_settings_topics": 0,
                "no_source_topics": 0,
                "estimated_success_rate": 0.0,
                "total_probed": 0,
                "passed_all_filters": 0,
                "primary_bottleneck": "",
                "message": "No study topics are available to validate.",
            }

        can_count = 0
        blocked_count = 0
        none_count = 0
        agg_total_probed = 0
        agg_passed_relevance = 0
        agg_passed_duration = 0
        agg_passed_clip = 0
        agg_passed_all = 0
        for material_id in material_ids:
            try:
                result = _probe_material_viability(
                    conn,
                    material_id=material_id,
                    concept_id=None,
                    creative_commons_only=payload.creative_commons_only,
                    fast_mode=payload.generation_mode == "fast",
                    min_relevance=min_relevance,
                    video_pool_mode=safe_video_pool_mode,
                    preferred_video_duration=safe_video_duration_pref,
                    target_clip_duration_sec=safe_target_clip_duration_sec,
                    target_clip_duration_min_sec=safe_target_clip_min_sec,
                    target_clip_duration_max_sec=safe_target_clip_max_sec,
                )
            except YouTubeApiRequestError as e:
                raise HTTPException(status_code=502, detail=str(e)) from e
            agg_total_probed += result.total_probed
            agg_passed_relevance += result.passed_relevance
            agg_passed_duration += result.passed_duration_pref
            agg_passed_clip += result.passed_clip_range
            agg_passed_all += result.passed_all
            if result.can_generate:
                can_count += 1
            elif result.blocked_by_settings:
                blocked_count += 1
            else:
                none_count += 1

    checked_count = len(material_ids)
    estimated_rate = round(agg_passed_all / agg_total_probed, 4) if agg_total_probed > 0 else 0.0
    bottleneck = _determine_primary_bottleneck(
        agg_total_probed, agg_passed_relevance, agg_passed_duration, agg_passed_clip,
    )
    base = {
        "topics_checked": checked_count,
        "topics_can_generate": can_count,
        "blocked_by_settings_topics": blocked_count,
        "no_source_topics": none_count,
        "estimated_success_rate": estimated_rate,
        "total_probed": agg_total_probed,
        "passed_all_filters": agg_passed_all,
        "primary_bottleneck": bottleneck,
    }

    rate_pct = round(estimated_rate * 100)
    if can_count > 0:
        bottleneck_hint = f" Bottleneck: {bottleneck}." if bottleneck and bottleneck != "no_source" and can_count < checked_count else ""
        return {
            **base,
            "can_generate_any": True,
            "message": f"Estimated success rate: {rate_pct}% ({agg_passed_all}/{agg_total_probed} probed reels pass). {can_count}/{checked_count} topics can generate.{bottleneck_hint}",
        }

    if blocked_count > 0:
        bottleneck_hint = f" Primary bottleneck: {bottleneck}." if bottleneck and bottleneck != "no_source" else ""
        return {
            **base,
            "can_generate_any": False,
            "message": f"Estimated success rate: 0% (0/{agg_total_probed} probed reels pass). Settings too strict for {blocked_count}/{checked_count} topics.{bottleneck_hint}",
        }

    return {
        **base,
        "can_generate_any": False,
        "message": f"Estimated success rate: 0%. No matching source videos found across {checked_count} topics.",
    }


@app.get("/api/feed", response_model=FeedResponse)
def feed(
    material_id: str,
    page: int = 1,
    limit: int = 5,
    autofill: bool = True,
    prefetch: int = 7,
    creative_commons_only: bool = False,
    generation_mode: Literal["slow", "fast"] = "fast",
    min_relevance: float | None = None,
    video_pool_mode: Literal["short-first", "balanced", "long-form"] = "short-first",
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any",
    target_clip_duration_sec: int = DEFAULT_TARGET_CLIP_DURATION_SEC,
    target_clip_duration_min_sec: int | None = None,
    target_clip_duration_max_sec: int | None = None,
):
    if page < 1:
        page = 1
    if limit < 1:
        limit = 1
    if limit > 25:
        limit = 25
    if prefetch < 0:
        prefetch = 0
    if prefetch > 30:
        prefetch = 30

    with get_conn() as conn:
        material = fetch_all(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")

        fast_mode = generation_mode == "fast"
        if SERVERLESS_MODE:
            # Hosted/serverless: force fast retrieval profile to avoid request timeouts.
            fast_mode = True
            prefetch = min(prefetch, 6)
        safe_min_relevance = _normalize_min_relevance(min_relevance)
        safe_video_pool_mode = _normalize_video_pool_mode(video_pool_mode)
        safe_video_duration_pref = _normalize_preferred_video_duration(preferred_video_duration)
        safe_target_clip_duration_sec, safe_target_clip_min_sec, safe_target_clip_max_sec = _resolve_target_clip_duration_bounds(
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        ranked = reel_service.ranked_feed(conn, material_id, fast_mode=fast_mode)
        filtered_ranked = _filter_reels_by_min_relevance(ranked, safe_min_relevance)
        filtered_ranked = _filter_reels_by_video_preferences(
            filtered_ranked,
            preferred_video_duration=safe_video_duration_pref,
            target_clip_duration_sec=safe_target_clip_duration_sec,
            target_clip_duration_min_sec=safe_target_clip_min_sec,
            target_clip_duration_max_sec=safe_target_clip_max_sec,
        )
        # Auto-expand the feed while users scroll so we can keep serving fresh reels.
        if autofill:
            if SERVERLESS_MODE:
                target_total = page * limit + prefetch + 2
                max_attempts = 1
                max_batch = 6
            else:
                target_total = page * limit + prefetch + (10 if fast_mode else 6)
                max_attempts = 5 if fast_mode else 4
                max_batch = 18 if fast_mode else 14
            attempts = 0
            while len(filtered_ranked) < target_total and attempts < max_attempts:
                need = max(1, target_total - len(filtered_ranked))
                try:
                    generated = reel_service.generate_reels(
                        conn,
                        material_id=material_id,
                        concept_id=None,
                        num_reels=min(max_batch, max(need + 4, limit + 4)),
                        creative_commons_only=creative_commons_only,
                        fast_mode=fast_mode,
                        video_pool_mode=safe_video_pool_mode,
                        preferred_video_duration=safe_video_duration_pref,
                        target_clip_duration_sec=safe_target_clip_duration_sec,
                        target_clip_duration_min_sec=safe_target_clip_min_sec,
                        target_clip_duration_max_sec=safe_target_clip_max_sec,
                    )
                except YouTubeApiRequestError:
                    break
                attempts += 1
                if not generated:
                    break
                ranked = reel_service.ranked_feed(conn, material_id, fast_mode=fast_mode)
                filtered_ranked = _filter_reels_by_min_relevance(ranked, safe_min_relevance)
                filtered_ranked = _filter_reels_by_video_preferences(
                    filtered_ranked,
                    preferred_video_duration=safe_video_duration_pref,
                    target_clip_duration_sec=safe_target_clip_duration_sec,
                    target_clip_duration_min_sec=safe_target_clip_min_sec,
                    target_clip_duration_max_sec=safe_target_clip_max_sec,
                )
        else:
            filtered_ranked = _filter_reels_by_min_relevance(ranked, safe_min_relevance)
            filtered_ranked = _filter_reels_by_video_preferences(
                filtered_ranked,
                preferred_video_duration=safe_video_duration_pref,
                target_clip_duration_sec=safe_target_clip_duration_sec,
                target_clip_duration_min_sec=safe_target_clip_min_sec,
                target_clip_duration_max_sec=safe_target_clip_max_sec,
            )

    total = len(filtered_ranked)
    start = (page - 1) * limit
    end = start + limit
    reels = filtered_ranked[start:end]

    return {"page": page, "limit": limit, "total": total, "reels": reels}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    answer = material_intelligence_service.chat_assistant(
        message=payload.message,
        topic=payload.topic,
        text=payload.text,
        history=history,
    )
    return {"answer": answer}


@app.post("/api/reels/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest):
    with get_conn() as conn:
        exists = fetch_all(conn, "SELECT id FROM reels WHERE id = ?", (payload.reel_id,))
        if not exists:
            raise HTTPException(status_code=404, detail="reel_id not found")

        reel_service.record_feedback(
            conn,
            reel_id=payload.reel_id,
            helpful=payload.helpful,
            confusing=payload.confusing,
            rating=payload.rating,
            saved=payload.saved,
        )

    return {"status": "ok", "reel_id": payload.reel_id}


@app.get("/api/community/reels/duration")
def get_community_reel_duration(source_url: str):
    normalized_url = source_url.strip()
    if not normalized_url:
        raise HTTPException(status_code=400, detail="source_url is required.")
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="source_url must be an absolute http(s) URL.")

    duration_sec = _resolve_community_reel_duration_sec(normalized_url)
    return {"duration_sec": duration_sec}


@app.get("/api/community/sets", response_model=CommunitySetsResponse)
def list_community_sets(limit: int = 160):
    safe_limit = max(1, min(limit, 300))
    with get_conn() as conn:
        rows = fetch_all(
            conn,
            """
            SELECT
                id,
                title,
                description,
                tags_json,
                reels_json,
                reel_count,
                curator,
                likes,
                learners,
                updated_label,
                thumbnail_url,
                featured,
                created_at
            FROM community_sets
            ORDER BY featured DESC, created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        )
    sets = [_serialize_community_set(row) for row in rows]
    return {"sets": sets}


@app.post("/api/community/sets", response_model=CommunitySetOut, status_code=201)
def create_community_set(payload: CommunitySetCreateRequest):
    title = payload.title.strip()
    description = payload.description.strip()
    thumbnail_url = payload.thumbnail_url.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Set name is required.")
    if len(description) < 18:
        raise HTTPException(status_code=400, detail="Description must be at least 18 characters.")
    if not thumbnail_url:
        raise HTTPException(status_code=400, detail="Thumbnail is required.")
    if not payload.reels:
        raise HTTPException(status_code=400, detail="Add at least one reel URL.")

    reels_payload: list[dict[str, object]] = []
    for reel in payload.reels:
        source_url = reel.source_url.strip()
        embed_url = reel.embed_url.strip()
        if not source_url or not embed_url:
            continue
        t_start_sec = _normalize_clip_seconds(reel.t_start_sec)
        t_end_sec = _normalize_clip_seconds(reel.t_end_sec)
        if t_end_sec is not None:
            start_for_validation = t_start_sec if t_start_sec is not None else 0.0
            if t_end_sec <= start_for_validation:
                raise HTTPException(status_code=400, detail="Clip end time must be greater than start time.")
        reels_payload.append(
            {
                "id": str(uuid.uuid4()),
                "platform": reel.platform,
                "source_url": source_url,
                "embed_url": embed_url,
                **({"t_start_sec": t_start_sec} if t_start_sec is not None else {}),
                **({"t_end_sec": t_end_sec} if t_end_sec is not None else {}),
            }
        )
    if not reels_payload:
        raise HTTPException(status_code=400, detail="No valid reels found in request.")

    tags = _normalize_community_tags(payload.tags)
    curator = (payload.curator or "").strip() or "Community member"
    set_id = f"user-set-{uuid.uuid4()}"
    created_at = now_iso()
    updated_label = "Updated just now"

    with get_conn() as conn:
        upsert(
            conn,
            "community_sets",
            {
                "id": set_id,
                "title": title,
                "description": description,
                "tags_json": dumps_json(tags),
                "reels_json": dumps_json(reels_payload),
                "reel_count": len(reels_payload),
                "curator": curator,
                "likes": 0,
                "learners": 1,
                "updated_label": updated_label,
                "thumbnail_url": thumbnail_url,
                "featured": 0,
                "created_at": created_at,
                "updated_at": created_at,
            },
        )
        created_rows = fetch_all(
            conn,
            """
            SELECT
                id,
                title,
                description,
                tags_json,
                reels_json,
                reel_count,
                curator,
                likes,
                learners,
                updated_label,
                thumbnail_url,
                featured
            FROM community_sets
            WHERE id = ?
            LIMIT 1
            """,
            (set_id,),
        )

    if not created_rows:
        raise HTTPException(status_code=500, detail="Failed to persist community set.")

    return _serialize_community_set(created_rows[0])


@app.put("/api/community/sets/{set_id}", response_model=CommunitySetOut)
def update_community_set(set_id: str, payload: CommunitySetUpdateRequest):
    normalized_set_id = set_id.strip()
    if not normalized_set_id:
        raise HTTPException(status_code=400, detail="set_id is required.")
    if not normalized_set_id.startswith("user-set-"):
        raise HTTPException(status_code=403, detail="Only user-created sets can be edited.")

    title = payload.title.strip()
    description = payload.description.strip()
    thumbnail_url = payload.thumbnail_url.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Set name is required.")
    if len(description) < 18:
        raise HTTPException(status_code=400, detail="Description must be at least 18 characters.")
    if not thumbnail_url:
        raise HTTPException(status_code=400, detail="Thumbnail is required.")
    if not payload.reels:
        raise HTTPException(status_code=400, detail="Add at least one reel URL.")

    reels_payload: list[dict[str, object]] = []
    for reel in payload.reels:
        source_url = reel.source_url.strip()
        embed_url = reel.embed_url.strip()
        if not source_url or not embed_url:
            continue
        t_start_sec = _normalize_clip_seconds(reel.t_start_sec)
        t_end_sec = _normalize_clip_seconds(reel.t_end_sec)
        if t_end_sec is not None:
            start_for_validation = t_start_sec if t_start_sec is not None else 0.0
            if t_end_sec <= start_for_validation:
                raise HTTPException(status_code=400, detail="Clip end time must be greater than start time.")
        reels_payload.append(
            {
                "id": str(uuid.uuid4()),
                "platform": reel.platform,
                "source_url": source_url,
                "embed_url": embed_url,
                **({"t_start_sec": t_start_sec} if t_start_sec is not None else {}),
                **({"t_end_sec": t_end_sec} if t_end_sec is not None else {}),
            }
        )
    if not reels_payload:
        raise HTTPException(status_code=400, detail="No valid reels found in request.")

    tags = _normalize_community_tags(payload.tags)
    updated_at = now_iso()
    updated_label = "Updated just now"

    with get_conn() as conn:
        existing_rows = fetch_all(
            conn,
            """
            SELECT
                id,
                curator,
                likes,
                learners,
                featured,
                created_at
            FROM community_sets
            WHERE id = ?
            LIMIT 1
            """,
            (normalized_set_id,),
        )
        if not existing_rows:
            raise HTTPException(status_code=404, detail="Community set not found.")
        existing = existing_rows[0]
        if _to_int(existing.get("featured"), 0) != 0:
            raise HTTPException(status_code=403, detail="Featured sets cannot be edited.")

        existing_curator = str(existing.get("curator") or "").strip()
        curator = (payload.curator or "").strip() or existing_curator or "Community member"
        created_at = str(existing.get("created_at") or "").strip() or updated_at

        upsert(
            conn,
            "community_sets",
            {
                "id": normalized_set_id,
                "title": title,
                "description": description,
                "tags_json": dumps_json(tags),
                "reels_json": dumps_json(reels_payload),
                "reel_count": len(reels_payload),
                "curator": curator,
                "likes": max(0, _to_int(existing.get("likes"), 0)),
                "learners": max(0, _to_int(existing.get("learners"), 1)),
                "updated_label": updated_label,
                "thumbnail_url": thumbnail_url,
                "featured": 0,
                "created_at": created_at,
                "updated_at": updated_at,
            },
        )
        updated_rows = fetch_all(
            conn,
            """
            SELECT
                id,
                title,
                description,
                tags_json,
                reels_json,
                reel_count,
                curator,
                likes,
                learners,
                updated_label,
                thumbnail_url,
                featured
            FROM community_sets
            WHERE id = ?
            LIMIT 1
            """,
            (normalized_set_id,),
        )

    if not updated_rows:
        raise HTTPException(status_code=500, detail="Failed to persist community set changes.")

    return _serialize_community_set(updated_rows[0])


def _register_non_api_route_aliases() -> None:
    # Some deployments mount this ASGI app under `/api` already. Registering
    # unprefixed aliases prevents 404s when upstream strips that prefix.
    existing: set[tuple[str, frozenset[str]]] = set()
    for route in app.routes:
        if isinstance(route, APIRoute):
            existing.add((route.path, frozenset(route.methods or set())))

    for route in list(app.routes):
        if not isinstance(route, APIRoute):
            continue
        if not route.path.startswith("/api/"):
            continue
        alias_path = route.path[len("/api"):]
        methods = frozenset(route.methods or set())
        key = (alias_path, methods)
        if key in existing:
            continue
        app.add_api_route(
            alias_path,
            route.endpoint,
            methods=list(methods),
            name=f"{route.name}__alias",
            include_in_schema=False,
            response_model=route.response_model,
            status_code=route.status_code,
            response_description=route.response_description,
            responses=route.responses,
            deprecated=route.deprecated,
            summary=route.summary,
            description=route.description,
            response_model_include=route.response_model_include,
            response_model_exclude=route.response_model_exclude,
            response_model_by_alias=route.response_model_by_alias,
            response_model_exclude_unset=route.response_model_exclude_unset,
            response_model_exclude_defaults=route.response_model_exclude_defaults,
            response_model_exclude_none=route.response_model_exclude_none,
        )
        existing.add(key)


_register_non_api_route_aliases()
