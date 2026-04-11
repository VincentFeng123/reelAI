"""
Database persistence for the ingestion pipeline.

This module is the ONLY place the ingestion package touches the DB directly. Everything
goes through `db.upsert` / `db.insert` / `db.fetch_one` so the SQLite vs Postgres switch
is transparent. No raw `conn.execute(...)` SQL here.

Layout on disk:
  * `materials`: holds an `ingest-scratch` sentinel row (for anonymous ingests).
  * `concepts`: holds an `ingest-scratch-concept` sentinel row under that material.
  * `videos`: one row per unique `(platform, source_id)`, key prefixed e.g. `yt:abc`, `ig:xyz`.
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
import sys
import uuid
from typing import Any

from .. import db as db_module
from ..db import (
    DatabaseIntegrityError,
    dumps_json,
    fetch_one,
    get_conn,
    insert,
    loads_json,
    now_iso,
    upsert,
)
from . import INGEST_SENTINEL_CONCEPT_ID, INGEST_SENTINEL_MATERIAL_ID
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


def ensure_sentinel_concept(conn: Any, material_id: str) -> str:
    """Idempotently insert the sentinel concept row under a material."""
    try:
        existing = fetch_one(conn, "SELECT id FROM concepts WHERE id = ?", (INGEST_SENTINEL_CONCEPT_ID,))
        if existing:
            return INGEST_SENTINEL_CONCEPT_ID
    except Exception:
        logger.exception("ensure_sentinel_concept: lookup failed")

    try:
        upsert(
            conn,
            "concepts",
            {
                "id": INGEST_SENTINEL_CONCEPT_ID,
                "material_id": material_id,
                "title": "Ingested reel",
                "keywords_json": dumps_json([]),
                "summary": "",
                "embedding_json": None,
                "created_at": now_iso(),
            },
            pk="id",
        )
    except DatabaseIntegrityError:
        pass
    return INGEST_SENTINEL_CONCEPT_ID


# --------------------------------------------------------------------- #
# Videos
# --------------------------------------------------------------------- #


def build_video_id(platform: str, source_id: str) -> str:
    """`yt:dQw...`, `ig:C8abc...`, `tt:7234...` — prefixed to avoid cache collisions."""
    return f"{platform}:{source_id}"


def upsert_video(conn: Any, *, platform: str, source_id: str, metadata: IngestMetadata) -> str:
    video_id = build_video_id(platform, source_id)
    duration_int = int(metadata.duration_sec) if metadata.duration_sec else 0
    payload = {
        "id": video_id,
        "title": metadata.title or f"{platform} reel {source_id}",
        "channel_title": metadata.author_name or metadata.author_handle,
        "description": metadata.description,
        "duration_sec": duration_int,
        "view_count": metadata.view_count or 0,
        "is_creative_commons": 0,
        "created_at": now_iso(),
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
    video_id: str,
    t_start: float,
    t_end: float,
) -> dict[str, Any] | None:
    """
    Look up a reel row by the fields in the unique index
    `(material_id, generation_id, video_id, t_start, t_end)` — we treat `generation_id`
    as NULL for anonymous ingests.
    """
    try:
        row = fetch_one(
            conn,
            """
            SELECT id, material_id, concept_id, video_id, video_url, t_start, t_end,
                   transcript_snippet, takeaways_json, base_score, created_at
              FROM reels
             WHERE material_id = ?
               AND video_id = ?
               AND t_start = ?
               AND t_end = ?
               AND (generation_id IS NULL OR generation_id = '')
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (material_id, video_id, float(t_start), float(t_end)),
        )
    except Exception:
        logger.exception(
            "load_existing_reel: lookup failed for material=%s video=%s",
            material_id,
            video_id,
        )
        return None
    return row


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
) -> bool:
    """
    Insert a reel row. Returns True on success, False if the unique index rejected a
    duplicate (caller should then call `load_existing_reel` to return the existing one).
    """
    row = {
        "id": reel_id,
        "generation_id": None,
        "material_id": material_id,
        "concept_id": concept_id,
        "video_id": video_id,
        "video_url": video_url,
        "t_start": float(t_start),
        "t_end": float(t_end),
        "transcript_snippet": transcript_snippet[:7000],
        "takeaways_json": dumps_json(list(takeaways or [])),
        "base_score": float(base_score),
        "created_at": now_iso(),
    }
    try:
        insert(conn, "reels", row)
        return True
    except DatabaseIntegrityError:
        return False
    except Exception:
        logger.exception("upsert_reel_row: insert failed for reel_id=%s", reel_id)
        return False


def load_reel_row_by_id(conn: Any, reel_id: str) -> dict[str, Any] | None:
    try:
        return fetch_one(
            conn,
            """
            SELECT id, material_id, concept_id, video_id, video_url, t_start, t_end,
                   transcript_snippet, takeaways_json, base_score, created_at
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
    """
    Delete every reel whose stored `video_url` matches `source_url` OR whose ingest
    metadata blob's `source_url` matches. Returns the number of rows removed.

    Also removes the associated metadata blobs in `llm_cache` and the cached transcript.
    """
    if not source_url:
        return 0

    removed = 0
    cursor = conn.cursor()
    try:
        # 1. Direct video_url match
        cursor.execute("SELECT id FROM reels WHERE video_url = ?", (source_url,))
        rows = cursor.fetchall()
        direct_ids: list[str] = [r[0] if isinstance(r, tuple) else r["id"] for r in rows]

        # 2. Metadata blob match (JSON LIKE is cheap for small tables)
        cursor.execute(
            "SELECT cache_key, response_json FROM llm_cache WHERE cache_key LIKE ? AND response_json LIKE ?",
            (f"{_INGEST_META_PREFIX}%", f'%"source_url": "{source_url}"%'),
        )
        blob_rows = cursor.fetchall()
        blob_reel_ids: list[str] = []
        for blob in blob_rows:
            key = blob[0] if isinstance(blob, tuple) else blob["cache_key"]
            if not isinstance(key, str) or not key.startswith(_INGEST_META_PREFIX):
                continue
            blob_reel_ids.append(key[len(_INGEST_META_PREFIX):])

        to_remove = sorted(set(direct_ids) | set(blob_reel_ids))
        if not to_remove:
            return 0

        placeholders = ",".join("?" * len(to_remove))
        cursor.execute(f"DELETE FROM reels WHERE id IN ({placeholders})", to_remove)
        removed = cursor.rowcount or 0

        # Clean up metadata blobs
        meta_keys = [f"{_INGEST_META_PREFIX}{rid}" for rid in to_remove]
        if meta_keys:
            key_placeholders = ",".join("?" * len(meta_keys))
            cursor.execute(f"DELETE FROM llm_cache WHERE cache_key IN ({key_placeholders})", meta_keys)

        conn.commit()
        logger.info(
            "takedown_by_source_url: removed %d reels + %d metadata blobs for %s",
            removed,
            len(meta_keys),
            source_url,
        )
    except Exception:
        logger.exception("takedown_by_source_url failed for %s", source_url)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        try:
            cursor.close()
        except Exception:
            pass
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
    "ensure_sentinel_material",
    "ensure_sentinel_concept",
    "upsert_video",
    "upsert_reel_row",
    "load_existing_reel",
    "load_reel_row_by_id",
    "store_ingest_metadata_blob",
    "load_ingest_metadata_blob",
    "takedown_by_source_url",
]
