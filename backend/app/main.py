import json
import os
import uuid
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import dumps_json, fetch_all, get_conn, init_db, now_iso, upsert
from .models import (
    ChatRequest,
    ChatResponse,
    CommunityReelOut,
    CommunitySetCreateRequest,
    CommunitySetOut,
    CommunitySetsResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedResponse,
    MaterialResponse,
    ReelsGenerateRequest,
    ReelsGenerateResponse,
)
from .services.embeddings import EmbeddingService
from .services.material_intelligence import MaterialIntelligenceService
from .services.parsers import ParseError, extract_text_from_file
from .services.reels import ReelService
from .services.storage import get_storage
from .services.text_utils import chunk_text, normalize_whitespace
from .services.youtube import YouTubeApiRequestError, YouTubeService

settings = get_settings()
app = FastAPI(title="StudyReels API", version="0.1.0")

allowed_origins = [
    settings.frontend_origin,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = get_storage()
embedding_service = EmbeddingService()
material_intelligence_service = MaterialIntelligenceService()
youtube_service = YouTubeService()
reel_service = ReelService(embedding_service=embedding_service, youtube_service=youtube_service)


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


def _normalize_min_relevance(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(-1.0, min(1.2, parsed))


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
            reels.append(
                CommunityReelOut(
                    id=reel_id,
                    platform=platform,
                    source_url=source_url,
                    embed_url=embed_url,
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

    material_id = str(uuid.uuid4())
    created_at = now_iso()

    with get_conn() as conn:
        concepts, objectives = material_intelligence_service.extract_concepts_and_objectives(
            conn,
            raw_text,
            subject_tag=subject_tag,
            max_concepts=12,
        )
        if objectives and not any(c["title"].lower() == "learning objectives" for c in concepts):
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
    with get_conn() as conn:
        material = fetch_all(conn, "SELECT id FROM materials WHERE id = ?", (payload.material_id,))
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")

        try:
            reels = reel_service.generate_reels(
                conn,
                material_id=payload.material_id,
                concept_id=payload.concept_id,
                num_reels=payload.num_reels,
                creative_commons_only=payload.creative_commons_only,
                fast_mode=payload.generation_mode == "fast",
            )
        except YouTubeApiRequestError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
    return {"reels": _filter_reels_by_min_relevance(reels, min_relevance)}


@app.get("/api/feed", response_model=FeedResponse)
def feed(
    material_id: str,
    page: int = 1,
    limit: int = 5,
    autofill: bool = True,
    prefetch: int = 7,
    creative_commons_only: bool = False,
    generation_mode: Literal["slow", "fast"] = "slow",
    min_relevance: float | None = None,
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
        safe_min_relevance = _normalize_min_relevance(min_relevance)
        ranked = reel_service.ranked_feed(conn, material_id, fast_mode=fast_mode)
        filtered_ranked = _filter_reels_by_min_relevance(ranked, safe_min_relevance)
        # Auto-expand the feed while users scroll so we can keep serving fresh reels.
        if autofill:
            target_total = page * limit + prefetch
            attempts = 0
            while len(filtered_ranked) < target_total and attempts < 3:
                need = max(1, target_total - len(filtered_ranked))
                try:
                    generated = reel_service.generate_reels(
                        conn,
                        material_id=material_id,
                        concept_id=None,
                        num_reels=min(12, need + 2),
                        creative_commons_only=creative_commons_only,
                        fast_mode=fast_mode,
                    )
                except YouTubeApiRequestError:
                    break
                attempts += 1
                if not generated:
                    break
                ranked = reel_service.ranked_feed(conn, material_id, fast_mode=fast_mode)
                filtered_ranked = _filter_reels_by_min_relevance(ranked, safe_min_relevance)
        else:
            filtered_ranked = _filter_reels_by_min_relevance(ranked, safe_min_relevance)

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

    reels_payload: list[dict[str, str]] = []
    for reel in payload.reels:
        source_url = reel.source_url.strip()
        embed_url = reel.embed_url.strip()
        if not source_url or not embed_url:
            continue
        reels_payload.append(
            {
                "id": str(uuid.uuid4()),
                "platform": reel.platform,
                "source_url": source_url,
                "embed_url": embed_url,
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
