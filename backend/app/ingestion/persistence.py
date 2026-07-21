"""
Database persistence for the ingestion pipeline.

This module is the ONLY place the ingestion package touches the DB directly. Everything
goes through `db.upsert` / `db.insert` / `db.fetch_one` so the SQLite vs Postgres switch
is transparent. No raw `conn.execute(...)` SQL here.

Layout on disk:
  * `materials`: holds an `ingest-scratch` sentinel row (for anonymous ingests).
  * `concepts`: holds an `ingest-scratch-concept` sentinel row under that material.
  * `videos`: one row per YouTube source ID, stored with an internal `yt:` prefix.
  * `reels`: one row per generated clip — the existing table, no migration.
  * `llm_cache` key `ingest_meta:{reel_id}`: JSON blob with the extended metadata
    (author_handle, hashtags, likes, location, audio_info, thumbnail_url, ...) that
    doesn't fit in the `reels` table. Not stored in `ranked_feed_cache` because that
    table has additional NOT NULL columns that would force us to synthesize values.

DMCA admin helper: `takedown_by_source_url(conn, source_url)` deletes every reel row
whose video_url matches (or whose ingest metadata blob records the same source URL).
Callable from a CLI: `python -m backend.app.ingestion.persistence --takedown <url>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
import uuid
from typing import Any

from ...concept_families import concept_family_identity_key
from ...concept_tokens import semantic_tokens
from .. import db as db_module
from ..db import (
    DatabaseIntegrityError,
    dumps_json,
    execute_modify,
    fetch_all,
    fetch_one,
    get_conn,
    insert,
    loads_json,
    now_iso,
    upsert,
)
from . import INGEST_SENTINEL_CONCEPT_ID, INGEST_SENTINEL_MATERIAL_ID
from .errors import InvalidReferenceError
from .logging_config import get_ingest_logger
from .models import IngestMetadata

logger: logging.Logger = get_ingest_logger(__name__)

_INGEST_META_PREFIX = "ingest_meta:"
# --------------------------------------------------------------------- #
# Sentinel materials / concepts
# --------------------------------------------------------------------- #


def ensure_sentinel_material(conn: Any) -> str:
    """Idempotently insert the sentinel material row used for anonymous ingests."""
    try:
        existing = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (INGEST_SENTINEL_MATERIAL_ID,))
        if existing:
            return INGEST_SENTINEL_MATERIAL_ID
    except Exception:
        logger.exception("ensure_sentinel_material: lookup failed")
        # Fall through to upsert — if the lookup failed, upsert will also fail, but at
        # least we'll get a consistent error.

    try:
        upsert(
            conn,
            "materials",
            {
                "id": INGEST_SENTINEL_MATERIAL_ID,
                "subject_tag": "ingest-scratch",
                "raw_text": "Anonymous URL ingests (ReelAI)",
                "source_type": "ingest",
                "source_path": None,
                "created_at": now_iso(),
            },
            pk="id",
        )
    except DatabaseIntegrityError:
        pass
    return INGEST_SENTINEL_MATERIAL_ID


def _scoped_sentinel_concept_id(material_id: str) -> str:
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"reelai:ingest-scratch:{material_id}").hex
    return f"ingest-scratch-concept:{suffix}"


def ensure_sentinel_concept(conn: Any, material_id: str) -> str:
    """Idempotently insert a deterministic scratch concept owned by ``material_id``."""
    material = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
    if not material:
        raise InvalidReferenceError("The supplied material does not exist.")

    concept_id = (
        INGEST_SENTINEL_CONCEPT_ID
        if material_id == INGEST_SENTINEL_MATERIAL_ID
        else _scoped_sentinel_concept_id(material_id)
    )
    existing = fetch_one(
        conn,
        "SELECT id, material_id FROM concepts WHERE id = ?",
        (concept_id,),
    )
    if existing and existing["material_id"] == material_id:
        return concept_id
    if existing:
        # Old versions could attach the global sentinel ID to an arbitrary material.
        # Leave that historical row untouched and use a scoped ID for the real global
        # sentinel material instead.
        if concept_id == INGEST_SENTINEL_CONCEPT_ID:
            concept_id = _scoped_sentinel_concept_id(material_id)
            existing = fetch_one(
                conn,
                "SELECT id, material_id FROM concepts WHERE id = ?",
                (concept_id,),
            )
            if existing and existing["material_id"] == material_id:
                return concept_id
        if existing:
            raise DatabaseIntegrityError("Scratch concept ID belongs to another material.")

    try:
        insert(
            conn,
            "concepts",
            {
                "id": concept_id,
                "material_id": material_id,
                "title": "Ingested reel",
                "keywords_json": dumps_json([]),
                "summary": "",
                "embedding_json": None,
                "created_at": now_iso(),
            },
        )
    except DatabaseIntegrityError:
        # Another worker may have inserted the same deterministic row.
        existing = fetch_one(
            conn,
            "SELECT id, material_id FROM concepts WHERE id = ?",
            (concept_id,),
        )
        if not existing or existing["material_id"] != material_id:
            raise
    return concept_id


def resolve_material_concept(
    conn: Any,
    *,
    material_id: str | None,
    concept_id: str | None,
) -> tuple[str, str]:
    """Resolve and validate caller-supplied material/concept references before writes."""
    concept = None
    if concept_id is not None:
        concept = fetch_one(
            conn,
            "SELECT id, material_id FROM concepts WHERE id = ?",
            (concept_id,),
        )
        if not concept:
            raise InvalidReferenceError("The supplied concept does not exist.")

    if material_id is not None:
        material = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
        if not material:
            raise InvalidReferenceError("The supplied material does not exist.")
    elif concept:
        material_id = str(concept["material_id"])
        material = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
        if not material:
            raise InvalidReferenceError("The supplied concept has no valid material.")
    else:
        material_id = ensure_sentinel_material(conn)

    if concept:
        if concept["material_id"] != material_id:
            raise InvalidReferenceError("The supplied concept does not belong to the material.")
        return material_id, str(concept["id"])

    return material_id, ensure_sentinel_concept(conn, material_id)


def normalize_clip_concept(value: object) -> tuple[str, str]:
    """Return a readable label and stable identity key for one Gemini facet."""
    title = " ".join(unicodedata.normalize("NFC", str(value or "")).split()).strip()
    title = title[:240]
    key = " ".join(semantic_tokens(title))
    return title, key


def normalize_clip_concept_family(value: object) -> tuple[str, str]:
    """Return a family key with possessives and ordinal spellings canonicalized."""
    title, _key = normalize_clip_concept(value)
    return title, concept_family_identity_key(title)


def ensure_clip_concept(
    conn: Any,
    *,
    material_id: str,
    title: str,
    semantic_identity: str | None = None,
) -> tuple[str, str, str]:
    """Reuse or create a material-scoped concept, optionally keyed by family."""
    clean_title, concept_key = normalize_clip_concept(title)
    if not clean_title or not concept_key:
        raise InvalidReferenceError("The clip concept is empty after normalization.")
    identity_title, semantic_key = normalize_clip_concept_family(
        semantic_identity
    )
    if semantic_identity is not None and not semantic_key:
        raise InvalidReferenceError("The clip concept family is empty after normalization.")

    material = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
    if not material:
        raise InvalidReferenceError("The supplied material does not exist.")

    if not semantic_key:
        existing_rows = fetch_all(
            conn,
            "SELECT id, title FROM concepts WHERE material_id = ? ORDER BY created_at, id",
            (material_id,),
        )
        for row in existing_rows:
            _existing_title, existing_key = normalize_clip_concept(row.get("title"))
            if existing_key == concept_key:
                return str(row["id"]), str(row["title"]), concept_key

    concept_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"reelai:clip-concept-family-v1:{material_id}:{semantic_key}"
                if semantic_key
                else f"reelai:clip-concept:{material_id}:{concept_key}"
            ),
        )
    )
    existing_id = fetch_one(
        conn,
        "SELECT id, material_id, title FROM concepts WHERE id = ?",
        (concept_id,),
    )
    if existing_id:
        if semantic_key:
            _existing_title, existing_key = normalize_clip_concept_family(
                existing_id.get("title")
            )
            if (
                existing_id.get("material_id") == material_id
                and existing_key == semantic_key
            ):
                return concept_id, str(existing_id["title"]), concept_key
            raise DatabaseIntegrityError(
                "Deterministic clip concept ID is already in use."
            )
        _existing_title, existing_key = normalize_clip_concept(
            existing_id.get("title")
        )
        if existing_id.get("material_id") == material_id and existing_key == concept_key:
            return concept_id, str(existing_id["title"]), concept_key
        raise DatabaseIntegrityError("Deterministic clip concept ID is already in use.")

    try:
        insert(
            conn,
            "concepts",
            {
                "id": concept_id,
                "material_id": material_id,
                "title": identity_title if semantic_key else clean_title,
                "keywords_json": dumps_json(
                    (semantic_key or concept_key).split()[:12]
                ),
                "summary": "",
                "embedding_json": None,
                "created_at": now_iso(),
            },
        )
    except DatabaseIntegrityError:
        # Concurrent workers derive the same primary key for the same normalized facet.
        existing_id = fetch_one(
            conn,
            "SELECT id, material_id, title FROM concepts WHERE id = ?",
            (concept_id,),
        )
        if not existing_id or existing_id.get("material_id") != material_id:
            raise
        normalizer = (
            normalize_clip_concept_family
            if semantic_key
            else normalize_clip_concept
        )
        _existing_title, existing_key = normalizer(existing_id.get("title"))
        if existing_key != (semantic_key or concept_key):
            raise
        return concept_id, str(existing_id["title"]), concept_key
    return concept_id, identity_title if semantic_key else clean_title, concept_key


# --------------------------------------------------------------------- #
# Videos
# --------------------------------------------------------------------- #


def build_video_id(platform: str, source_id: str) -> str:
    """`yt:dQw...`, `ig:C8abc...`, `tt:7234...` — prefixed to avoid cache collisions."""
    return f"{platform}:{source_id}"


def upsert_video(conn: Any, *, platform: str, source_id: str, metadata: IngestMetadata) -> str:
    video_id = build_video_id(platform, source_id)
    existing = fetch_one(conn, "SELECT * FROM videos WHERE id = ?", (video_id,)) or {}

    def nonblank(value: object, fallback: object = "") -> object:
        if isinstance(value, str):
            return value.strip() or fallback
        return value if value is not None else fallback

    duration_int = (
        int(metadata.duration_sec)
        if metadata.duration_sec is not None and metadata.duration_sec > 0
        else int(existing.get("duration_sec") or 0)
    )
    raw_view_count = metadata.view_count
    view_count = (
        int(raw_view_count)
        if isinstance(raw_view_count, int) and not isinstance(raw_view_count, bool)
        else int(existing.get("view_count") or 0)
    )
    payload = {
        "id": video_id,
        "title": nonblank(metadata.title, existing.get("title") or f"{platform} reel {source_id}"),
        "channel_title": nonblank(
            metadata.author_name or metadata.author_handle,
            existing.get("channel_title") or "",
        ),
        "channel_id": nonblank(metadata.channel_id, existing.get("channel_id") or ""),
        "description": nonblank(metadata.description, existing.get("description") or ""),
        "duration_sec": duration_int,
        "view_count": view_count,
        "is_creative_commons": int(existing.get("is_creative_commons") or 0),
        "provider": "youtube" if platform == "yt" else platform,
        "playback_url": nonblank(metadata.playback_url, existing.get("playback_url") or ""),
        "published_at": nonblank(metadata.upload_date_iso, existing.get("published_at") or ""),
        "source_url": nonblank(metadata.source_url, existing.get("source_url") or ""),
        "metadata_json": dumps_json(metadata.model_dump()),
        "created_at": str(existing.get("created_at") or now_iso()),
    }
    try:
        upsert(conn, "videos", payload, pk="id")
    except DatabaseIntegrityError:
        # Race condition: another worker inserted the same row. Ignore.
        pass
    return video_id


# --------------------------------------------------------------------- #
# Reels
# --------------------------------------------------------------------- #


def load_existing_reel(
    conn: Any,
    *,
    material_id: str,
    concept_id: str,
    video_id: str,
    t_start: float,
    t_end: float,
    generation_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Look up a reel row by the fields in the unique index
    `(material_id, generation_id, concept_id, video_id, t_start, t_end)`.
    Passing `generation_id=None` (the default) matches anonymous (NULL-or-empty) rows,
    preserving backward-compatible behavior.
    """
    try:
        row = fetch_one(
            conn,
            """
            SELECT id, material_id, concept_id, video_id, video_url, t_start, t_end,
                   transcript_snippet, takeaways_json, base_score, difficulty,
                   ai_summary, match_reason, informativeness, model_used,
                   quality_degraded, selected_cue_ids_json, search_context_json,
                   created_at
              FROM reels
             WHERE material_id = ?
               AND concept_id = ?
               AND video_id = ?
               AND t_start = ?
               AND t_end = ?
               AND COALESCE(generation_id, '') = COALESCE(?, '')
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (material_id, concept_id, video_id, float(t_start), float(t_end), generation_id),
        )
    except Exception:
        logger.exception(
            "load_existing_reel: lookup failed for material=%s video=%s",
            material_id,
            video_id,
        )
        return None
    return row


def load_reel_by_selection_candidate(
    conn: Any,
    *,
    material_id: str,
    concept_id: str,
    video_id: str,
    generation_id: str | None,
    selection_candidate_id: str,
) -> dict[str, Any] | None:
    """Find a previously stored acoustic state for the same selector candidate.

    Timestamps cannot identify the row because successful acoustic verification
    intentionally adjusts them. Candidate identity is scoped to one generation,
    concept, and source so promotion can update the existing reel safely.
    """
    candidate_id = str(selection_candidate_id or "").strip()
    if not candidate_id:
        return None
    rows = fetch_all(
        conn,
        """
        SELECT id, video_url, t_start, t_end, transcript_snippet,
               selected_cue_ids_json, search_context_json
          FROM reels
         WHERE material_id = ?
           AND concept_id = ?
           AND video_id = ?
           AND COALESCE(generation_id, '') = COALESCE(?, '')
         ORDER BY created_at DESC
        """,
        (material_id, concept_id, video_id, generation_id),
    )
    for row in rows:
        context = loads_json(str(row.get("search_context_json") or "{}"), {})
        if (
            isinstance(context, dict)
            and str(context.get("selection_candidate_id") or "").strip() == candidate_id
        ):
            return row
    return None


def update_reel_boundary_state(
    conn: Any,
    *,
    reel_id: str,
    video_url: str,
    t_start: float,
    t_end: float,
    transcript_snippet: str,
    selected_cue_ids: list[str],
    search_context: dict[str, Any],
    expected_search_context_json: str | None = None,
) -> bool:
    """Promote/defer a candidate in place without changing its public reel id."""
    conditional = (
        " AND search_context_json = ?"
        if expected_search_context_json is not None
        else ""
    )
    params: tuple[Any, ...] = (
        video_url,
        float(t_start),
        float(t_end),
        str(transcript_snippet or "")[:7000],
        dumps_json(list(selected_cue_ids or [])),
        dumps_json(dict(search_context or {})),
        reel_id,
    )
    if expected_search_context_json is not None:
        params = (*params, expected_search_context_json)
    changed = execute_modify(
        conn,
        f"""
        UPDATE reels
           SET video_url = ?,
               t_start = ?,
               t_end = ?,
               transcript_snippet = ?,
               selected_cue_ids_json = ?,
               search_context_json = ?
         WHERE id = ?{conditional}
        """,
        params,
    )
    return bool(changed)


def upsert_reel_row(
    conn: Any,
    *,
    reel_id: str,
    material_id: str,
    concept_id: str,
    video_id: str,
    video_url: str,
    t_start: float,
    t_end: float,
    transcript_snippet: str,
    takeaways: list[str],
    base_score: float = 1.0,
    generation_id: str | None = None,
    difficulty: float | None = None,
    ai_summary: str = "",
    match_reason: str = "",
    informativeness: float | None = None,
    model_used: str = "",
    quality_degraded: bool = False,
    selected_cue_ids: list[str] | None = None,
    search_context: dict[str, Any] | None = None,
) -> bool:
    """
    Insert a reel row. Returns True on success, False if the unique index rejected a
    duplicate (caller should then call `load_existing_reel` to return the existing one).
    """
    row = {
        "id": reel_id,
        "generation_id": generation_id,
        "material_id": material_id,
        "concept_id": concept_id,
        "video_id": video_id,
        "video_url": video_url,
        "t_start": float(t_start),
        "t_end": float(t_end),
        "transcript_snippet": transcript_snippet[:7000],
        "takeaways_json": dumps_json(list(takeaways or [])),
        "base_score": float(base_score),
        "difficulty": difficulty,
        "ai_summary": str(ai_summary or "")[:700],
        "match_reason": str(match_reason or "")[:700],
        "informativeness": informativeness,
        "model_used": str(model_used or "")[:160],
        "quality_degraded": 1 if quality_degraded else 0,
        "selected_cue_ids_json": dumps_json(list(selected_cue_ids or [])),
        "search_context_json": dumps_json(dict(search_context or {})),
        "created_at": now_iso(),
    }
    try:
        insert(conn, "reels", row)
        return True
    except DatabaseIntegrityError:
        return False


def load_reel_row_by_id(conn: Any, reel_id: str) -> dict[str, Any] | None:
    try:
        return fetch_one(
            conn,
            """
            SELECT id, material_id, concept_id, video_id, video_url, t_start, t_end,
                   transcript_snippet, takeaways_json, base_score, difficulty,
                   ai_summary, match_reason, informativeness, model_used,
                   quality_degraded, selected_cue_ids_json, search_context_json,
                   created_at
              FROM reels
             WHERE id = ?
            """,
            (reel_id,),
        )
    except Exception:
        logger.exception("load_reel_row_by_id failed for %s", reel_id)
        return None


# --------------------------------------------------------------------- #
# Ingest metadata blob (llm_cache)
# --------------------------------------------------------------------- #


def store_ingest_metadata_blob(conn: Any, *, reel_id: str, metadata: IngestMetadata) -> None:
    key = f"{_INGEST_META_PREFIX}{reel_id}"
    payload = metadata.model_dump()
    try:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": key,
                "response_json": dumps_json(payload),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
    except DatabaseIntegrityError:
        pass
    except Exception:
        logger.exception("store_ingest_metadata_blob failed for %s", reel_id)


def load_ingest_metadata_blob(conn: Any, *, reel_id: str) -> dict[str, Any] | None:
    key = f"{_INGEST_META_PREFIX}{reel_id}"
    try:
        row = fetch_one(
            conn,
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (key,),
        )
    except Exception:
        logger.exception("load_ingest_metadata_blob failed for %s", reel_id)
        return None
    if not row or not row.get("response_json"):
        return None
    data = loads_json(row["response_json"], default=None)
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------- #
# DMCA takedown admin helper
# --------------------------------------------------------------------- #


def takedown_by_source_url(conn: Any, source_url: str) -> int:
    """Transactionally tombstone and purge every artifact for a YouTube video.

    The caller owns the transaction. Any failure propagates so an API/CLI cannot
    report success before the surrounding context manager commits.
    """
    from ..clip_engine.metadata import extract_video_id

    clean_source_url = str(source_url or "").strip()
    bare_video_id = extract_video_id(clean_source_url)
    if not bare_video_id:
        raise ValueError("A valid YouTube video URL is required for takedown.")

    canonical_url = f"https://www.youtube.com/watch?v={bare_video_id}"
    prefixed_video_id = build_video_id("yt", bare_video_id)
    timestamp = now_iso()
    existing_tombstone = fetch_one(
        conn,
        "SELECT created_at FROM blocked_video_tombstones WHERE video_id = ?",
        (bare_video_id,),
    ) or {}
    upsert(
        conn,
        "blocked_video_tombstones",
        {
            "video_id": bare_video_id,
            "canonical_url": canonical_url,
            "source_url": clean_source_url,
            "reason": "takedown",
            "created_at": str(existing_tombstone.get("created_at") or timestamp),
            "updated_at": timestamp,
        },
        pk="video_id",
    )

    reel_rows = fetch_all(
        conn,
        "SELECT id, material_id, generation_id FROM reels WHERE video_id IN (?, ?)",
        (prefixed_video_id, bare_video_id),
    )
    direct_ids = {str(row.get("id") or "") for row in reel_rows if row.get("id")}

    # Include legacy rows whose only reliable source identity lives in metadata.
    blob_rows = fetch_all(
        conn,
        "SELECT cache_key, response_json FROM llm_cache WHERE cache_key LIKE ?",
        (f"{_INGEST_META_PREFIX}%",),
    )
    for blob in blob_rows:
        key = str(blob.get("cache_key") or "")
        metadata = loads_json(blob.get("response_json"), default={})
        if not key.startswith(_INGEST_META_PREFIX) or not isinstance(metadata, dict):
            continue
        if extract_video_id(str(metadata.get("source_url") or "")) == bare_video_id:
            direct_ids.add(key[len(_INGEST_META_PREFIX) :])

    to_remove = sorted(reel_id for reel_id in direct_ids if reel_id)
    affected_material_ids = {
        str(row.get("material_id") or "") for row in reel_rows if row.get("material_id")
    }
    affected_generation_ids = {
        str(row.get("generation_id") or "") for row in reel_rows if row.get("generation_id")
    }

    if to_remove:
        placeholders = ", ".join(["?"] * len(to_remove))
        question_rows = fetch_all(
            conn,
            f"SELECT id FROM reel_assessment_questions WHERE reel_id IN ({placeholders})",
            to_remove,
        )
        question_ids = [str(row.get("id") or "") for row in question_rows if row.get("id")]
        if question_ids:
            question_placeholders = ", ".join(["?"] * len(question_ids))
            session_rows = fetch_all(
                conn,
                "SELECT DISTINCT session_id FROM assessment_session_questions "
                f"WHERE question_id IN ({question_placeholders})",
                question_ids,
            )
            session_ids = [
                str(row.get("session_id") or "")
                for row in session_rows
                if row.get("session_id")
            ]
            if session_ids:
                session_placeholders = ", ".join(["?"] * len(session_ids))
                execute_modify(
                    conn,
                    f"DELETE FROM assessment_attempts WHERE session_id IN ({session_placeholders})",
                    session_ids,
                )
                execute_modify(
                    conn,
                    f"DELETE FROM assessment_concept_outcomes WHERE session_id IN ({session_placeholders})",
                    session_ids,
                )
                execute_modify(
                    conn,
                    f"DELETE FROM assessment_session_questions WHERE session_id IN ({session_placeholders})",
                    session_ids,
                )
                execute_modify(
                    conn,
                    f"DELETE FROM assessment_sessions WHERE id IN ({session_placeholders})",
                    session_ids,
                )
            execute_modify(
                conn,
                f"DELETE FROM assessment_attempts WHERE question_id IN ({question_placeholders})",
                question_ids,
            )
            execute_modify(
                conn,
                f"DELETE FROM assessment_session_questions WHERE question_id IN ({question_placeholders})",
                question_ids,
            )
            execute_modify(
                conn,
                f"DELETE FROM reel_assessment_questions WHERE id IN ({question_placeholders})",
                question_ids,
            )

        for table, column in (
            ("assessment_concept_outcomes", "source_reel_id"),
            ("learner_reel_progress", "reel_id"),
            ("reel_feedback", "reel_id"),
        ):
            execute_modify(
                conn,
                f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
                to_remove,
            )
        removed = execute_modify(
            conn,
            f"DELETE FROM reels WHERE id IN ({placeholders})",
            to_remove,
        )
        meta_keys = [f"{_INGEST_META_PREFIX}{reel_id}" for reel_id in to_remove]
        meta_placeholders = ", ".join(["?"] * len(meta_keys))
        execute_modify(
            conn,
            f"DELETE FROM llm_cache WHERE cache_key IN ({meta_placeholders})",
            meta_keys,
        )
    else:
        removed = 0

    # Remove source and retrieval artifacts before the video row (FK-safe).
    for table in (
        "retrieval_outcomes",
        "retrieval_candidates",
        "transcript_chunks",
    ):
        execute_modify(
            conn,
            f"DELETE FROM {table} WHERE video_id IN (?, ?)",
            (prefixed_video_id, bare_video_id),
        )
    execute_modify(
        conn,
        "DELETE FROM transcript_artifacts WHERE video_id IN (?, ?)",
        (prefixed_video_id, bare_video_id),
    )
    execute_modify(
        conn,
        "DELETE FROM transcript_cache WHERE video_id IN (?, ?)",
        (prefixed_video_id, bare_video_id),
    )
    execute_modify(
        conn,
        "DELETE FROM video_liveness_cache WHERE video_id IN (?, ?)",
        (prefixed_video_id, bare_video_id),
    )
    execute_modify(
        conn,
        "DELETE FROM videos WHERE id IN (?, ?)",
        (prefixed_video_id, bare_video_id),
    )

    # A tombstoned ID must not be rediscovered from any pre-takedown provider
    # page. Takedowns are rare, so clearing the small search caches is safer and
    # simpler than dialect-specific JSON inspection.
    execute_modify(conn, "DELETE FROM supadata_search_cache")
    execute_modify(conn, "DELETE FROM search_cache")

    if affected_material_ids:
        material_placeholders = ", ".join(["?"] * len(affected_material_ids))
        execute_modify(
            conn,
            f"DELETE FROM ranked_feed_cache WHERE material_id IN ({material_placeholders})",
            sorted(affected_material_ids),
        )
        execute_modify(
            conn,
            "UPDATE reel_generation_jobs SET cancel_requested = 1, updated_at = ? "
            f"WHERE material_id IN ({material_placeholders}) AND status IN ('queued', 'running')",
            (timestamp, *sorted(affected_material_ids)),
        )
    for generation_id in affected_generation_ids:
        count_row = fetch_one(
            conn,
            "SELECT COUNT(*) AS reel_count FROM reels WHERE generation_id = ?",
            (generation_id,),
        ) or {}
        execute_modify(
            conn,
            "UPDATE reel_generations SET reel_count = ? WHERE id = ?",
            (int(count_row.get("reel_count") or 0), generation_id),
        )

    logger.info(
        "takedown_by_source_url: tombstoned %s and removed %d reels",
        bare_video_id,
        removed,
    )
    return removed


# --------------------------------------------------------------------- #
# CLI entry point for takedowns
# --------------------------------------------------------------------- #


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ReelAI ingestion DB admin helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    takedown = sub.add_parser("takedown", help="Delete all reels for a given source URL")
    takedown.add_argument("url", help="The source URL to remove")

    lookup = sub.add_parser("lookup", help="Print the ingest metadata blob for a reel id")
    lookup.add_argument("reel_id")

    args = parser.parse_args(argv)

    if args.command == "takedown":
        # Opt-in path: ensure the DB is ready before we touch it.
        db_module.init_db()
        with get_conn(transactional=True) as conn:
            count = takedown_by_source_url(conn, args.url)
        print(json.dumps({"status": "ok", "removed": count, "source_url": args.url}, indent=2))
        return 0

    if args.command == "lookup":
        db_module.init_db()
        with get_conn() as conn:
            blob = load_ingest_metadata_blob(conn, reel_id=args.reel_id)
        print(json.dumps(blob or {}, indent=2, default=str))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_cli_main(sys.argv[1:]))


__all__ = [
    "build_video_id",
    "ensure_clip_concept",
    "ensure_sentinel_material",
    "ensure_sentinel_concept",
    "normalize_clip_concept",
    "resolve_material_concept",
    "upsert_video",
    "upsert_reel_row",
    "load_existing_reel",
    "load_reel_by_selection_candidate",
    "update_reel_boundary_state",
    "load_reel_row_by_id",
    "store_ingest_metadata_blob",
    "load_ingest_metadata_blob",
    "takedown_by_source_url",
]
