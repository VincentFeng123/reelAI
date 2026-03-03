import hashlib
import json
import re
import sqlite3
import uuid
from typing import Any

import numpy as np
from openai import OpenAI

from ..config import get_settings
from ..db import dumps_json, fetch_all, fetch_one, now_iso, upsert
from .concepts import build_takeaways
from .segmenter import (
    SegmentMatch,
    TranscriptChunk,
    chunk_transcript,
    lexical_overlap_score,
    normalize_terms,
    select_segments,
)


class ReelService:
    GENERIC_CONTEXT_TERMS = {
        "basics",
        "basic",
        "beginner",
        "beginners",
        "concept",
        "concepts",
        "course",
        "crash",
        "definition",
        "definitions",
        "example",
        "examples",
        "explained",
        "explainer",
        "explanation",
        "fundamental",
        "fundamentals",
        "guide",
        "introduction",
        "intro",
        "learn",
        "learning",
        "lesson",
        "lessons",
        "overview",
        "practice",
        "problem",
        "problems",
        "shorts",
        "study",
        "tutorial",
        "video",
    }
    OFF_TOPIC_PHRASES: dict[str, float] = {
        "motivational speech": 0.24,
        "inspirational speech": 0.22,
        "inspirational video": 0.16,
        "success motivation": 0.15,
        "whatsapp status": 0.2,
        "motivation status": 0.2,
        "quotes shorts": 0.16,
        "music video": 0.2,
        "lyrical video": 0.12,
        "reaction video": 0.1,
    }
    OFF_TOPIC_TOKENS = {
        "affirmation",
        "compilation",
        "edit",
        "grindset",
        "lyrics",
        "manifest",
        "meme",
        "motivation",
        "motivational",
        "podcast",
        "prank",
        "quotes",
        "reaction",
        "sigma",
        "status",
        "vlog",
    }

    def __init__(self, embedding_service, youtube_service) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service
        self.youtube_service = youtube_service
        self.chat_model = settings.openai_chat_model
        self.openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate_reels(
        self,
        conn,
        material_id: str,
        concept_id: str | None,
        num_reels: int,
        creative_commons_only: bool,
        fast_mode: bool = False,
    ) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = (material_id,)
        concept_where = "WHERE material_id = ?"
        if concept_id:
            concept_where += " AND id = ?"
            params = (material_id, concept_id)

        concepts = fetch_all(
            conn,
            (
                "SELECT id, material_id, title, keywords_json, summary, embedding_json, created_at "
                f"FROM concepts {concept_where} ORDER BY created_at ASC"
            ),
            params,
        )
        if not concepts:
            return []
        concepts = self._order_concepts(conn, material_id, concepts)
        if fast_mode:
            concepts = concepts[:4]

        existing_clip_keys = {
            self._clip_key(
                str(r.get("video_id") or ""),
                float(r.get("t_start") or 0),
                float(r.get("t_end") or 0),
            )
            for r in fetch_all(
                conn,
                "SELECT video_id, t_start, t_end FROM reels WHERE material_id = ?",
                (material_id,),
            )
            if r.get("video_id")
        }
        existing_video_counts = {
            str(r["video_id"]): int(r["reel_count"] or 0)
            for r in fetch_all(
                conn,
                "SELECT video_id, COUNT(*) AS reel_count FROM reels WHERE material_id = ? GROUP BY video_id",
                (material_id,),
            )
            if r.get("video_id")
        }
        generated_video_counts: dict[str, int] = {}
        generated_clip_keys: set[str] = set()
        max_segments_per_video = 1 if fast_mode else 3

        generated: list[dict[str, Any]] = []
        material = fetch_one(conn, "SELECT subject_tag FROM materials WHERE id = ?", (material_id,))
        subject_tag = material["subject_tag"] if material else None
        material_context_terms = self._build_material_context_terms(concepts=concepts, subject_tag=subject_tag)

        for concept in concepts:
            concept_keywords = json.loads(concept["keywords_json"])
            concept_embedding = self._get_concept_embedding(conn, concept) if not fast_mode else None
            concept_terms = [concept["title"], *concept_keywords, concept.get("summary", "")]
            context_terms = self._context_terms_for_concept(concept_terms, material_context_terms)
            vague_topic = self._is_vague_concept(
                title=concept["title"],
                keywords=concept_keywords,
                summary=concept.get("summary", ""),
            )
            queries = self._build_query_variants(
                concept["title"],
                concept_keywords,
                subject_tag,
                context_terms=context_terms,
            )
            if fast_mode:
                queries = queries[:2]
            seen_video_ids: set[str] = set()

            for query in queries:
                # First pull short-form videos; then broaden to any duration for long-form clipping.
                durations = ("short", None)
                for duration in durations:
                    prefer_short_query = duration == "short"
                    videos = self.youtube_service.search_videos(
                        conn,
                        query=query,
                        max_results=4 if fast_mode else 12,
                        creative_commons_only=creative_commons_only,
                        video_duration=duration,
                    )

                    for video in videos:
                        if video["id"] in seen_video_ids:
                            continue
                        seen_video_ids.add(video["id"])
                        existing_for_video = existing_video_counts.get(video["id"], 0)
                        generated_for_video = generated_video_counts.get(video["id"], 0)
                        if existing_for_video + generated_for_video >= max_segments_per_video:
                            continue
                        video_duration = int(video.get("duration_sec") or 0)

                        self._upsert_video(conn, video)
                        video_relevance = self._score_text_relevance(
                            conn,
                            text=self._video_metadata_text(video),
                            concept_terms=concept_terms,
                            context_terms=context_terms,
                            concept_embedding=concept_embedding,
                            subject_tag=subject_tag,
                        )
                        if not self._passes_relevance_gate(
                            relevance=video_relevance,
                            require_context=bool(context_terms) and vague_topic,
                            fast_mode=fast_mode,
                        ):
                            continue
                        video_relevance["passes"] = True
                        use_full_short_clip = self._should_use_full_short_clip(
                            prefer_short_query=prefer_short_query,
                            video_duration_sec=video_duration,
                        )
                        transcript = self.youtube_service.get_transcript(conn, video["id"])
                        segments: list[SegmentMatch] = []

                        if transcript:
                            if fast_mode:
                                segments = self._fast_segments_from_transcript(
                                    transcript=transcript,
                                    concept_terms=concept_terms,
                                    max_segments=2,
                                )
                            else:
                                chunks, chunk_embeddings = self._load_or_create_transcript_chunks(conn, video["id"], transcript)
                                if chunks and len(chunk_embeddings) > 0:
                                    segments = select_segments(
                                        concept_embedding,
                                        chunk_embeddings,
                                        chunks,
                                        concept_terms=concept_terms,
                                        top_k=6 if vague_topic else 4,
                                    )
                                    if vague_topic:
                                        expanded = self._split_video_into_short_segments(
                                            concept_embedding=concept_embedding,
                                            chunk_embeddings=chunk_embeddings,
                                            chunks=chunks,
                                            concept_terms=concept_terms,
                                            max_segments=8,
                                        )
                                        segments = self._merge_unique_segments([*segments, *expanded], max_items=8)

                            if not segments:
                                segments = self._fallback_segments_from_transcript(transcript)
                        else:
                            if not use_full_short_clip:
                                # Long-form cutting requires transcript timestamps.
                                continue
                            metadata_segment = self._fallback_segment_from_video_metadata(video, concept_terms)
                            if metadata_segment:
                                segments = [metadata_segment]
                        if not segments:
                            continue
                        segment_candidates = self._rank_segments_by_relevance(
                            conn,
                            segments=segments,
                            concept_terms=concept_terms,
                            context_terms=context_terms,
                            concept_embedding=concept_embedding,
                            subject_tag=subject_tag,
                            require_context=bool(context_terms) and vague_topic,
                            fast_mode=fast_mode,
                        )
                        if not segment_candidates:
                            continue

                        for segment, segment_relevance in segment_candidates:
                            if use_full_short_clip:
                                clip_window = self._full_short_clip_window(video_duration)
                            elif transcript:
                                clip_window = self._refine_clip_window_from_transcript(
                                    transcript=transcript,
                                    proposed_start=segment.t_start,
                                    proposed_end=segment.t_end,
                                    video_duration_sec=video_duration,
                                )
                            else:
                                clip_window = self._normalize_clip_window(
                                    segment.t_start,
                                    segment.t_end,
                                    video_duration,
                                    min_len=15,
                                    max_len=60,
                                )
                            if not clip_window:
                                continue
                            start_sec, end_sec = clip_window
                            clip_key = self._clip_key(video["id"], start_sec, end_sec)
                            if clip_key in existing_clip_keys or clip_key in generated_clip_keys:
                                continue
                            relevance_context = self._merge_relevance_context(video_relevance, segment_relevance)
                            if not bool(relevance_context.get("passes", True)):
                                continue
                            reel = self._create_reel(
                                conn,
                                material_id=material_id,
                                concept=concept,
                                video=video,
                                segment=segment,
                                clip_window=clip_window,
                                transcript=transcript,
                                relevance_context=relevance_context,
                                fast_mode=fast_mode,
                            )
                            if not reel:
                                continue
                            generated.append(reel)
                            generated_clip_keys.add(clip_key)
                            generated_video_counts[video["id"]] = generated_video_counts.get(video["id"], 0) + 1

                            if len(generated) >= num_reels:
                                return generated
                            if existing_for_video + generated_video_counts.get(video["id"], 0) >= max_segments_per_video:
                                break

        return generated

    def _get_concept_embedding(self, conn, concept: dict[str, Any]) -> np.ndarray:
        if concept.get("embedding_json"):
            return np.array(json.loads(concept["embedding_json"]), dtype=np.float32)

        concept_text = (
            f"{concept['title']}. "
            f"Keywords: {' '.join(json.loads(concept['keywords_json']))}. "
            f"Summary: {concept['summary']}"
        )
        embedding = self.embedding_service.embed_texts(conn, [concept_text])[0]
        existing = fetch_one(
            conn,
            "SELECT material_id, created_at FROM concepts WHERE id = ?",
            (concept["id"],),
        )
        if not existing:
            raise ValueError(f"Concept not found: {concept['id']}")
        upsert(
            conn,
            "concepts",
            {
                "id": concept["id"],
                "material_id": existing["material_id"],
                "title": concept["title"],
                "keywords_json": concept["keywords_json"],
                "summary": concept["summary"],
                "embedding_json": dumps_json(embedding.tolist()),
                "created_at": existing["created_at"],
            },
        )
        return embedding

    def _build_query_variants(
        self,
        title: str,
        keywords: list[str],
        subject_tag: str | None,
        context_terms: list[str] | None = None,
    ) -> list[str]:
        short_keywords = [k.strip() for k in keywords if k.strip()][:3]
        disambiguators = [term for term in (context_terms or []) if term.strip()][:2]
        subject = subject_tag.strip() if subject_tag else ""
        core = " ".join(part for part in [subject, title] if part).strip()
        context_hint = " ".join(disambiguators).strip()
        keyword_hint = " ".join(short_keywords).strip()

        variants = [
            f"{core} {keyword_hint} {context_hint} lecture explanation",
            f"{core} {keyword_hint} {context_hint} educational tutorial",
            f"{core} {keyword_hint} full lecture",
            f"{core} {context_hint} worked examples",
            f"{core} {keyword_hint} study guide",
            f"{core} explained",
        ]
        seen = set()
        deduped: list[str] = []
        for v in variants:
            cleaned = " ".join(v.split()).strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _build_material_context_terms(
        self,
        concepts: list[dict[str, Any]],
        subject_tag: str | None,
        max_terms: int = 24,
    ) -> list[str]:
        scores: dict[str, float] = {}
        order: dict[str, int] = {}

        def add_terms(value: str, weight: float) -> None:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-']*", value.lower()):
                if len(token) < 4:
                    continue
                if token in self.GENERIC_CONTEXT_TERMS:
                    continue
                if token not in order:
                    order[token] = len(order)
                scores[token] = scores.get(token, 0.0) + weight

        for concept in concepts:
            add_terms(str(concept.get("title") or ""), 2.0)
            add_terms(str(concept.get("summary") or ""), 0.7)
            keywords_json = str(concept.get("keywords_json") or "[]")
            try:
                keywords = json.loads(keywords_json)
            except json.JSONDecodeError:
                keywords = []
            if not isinstance(keywords, list):
                keywords = []
            for kw in keywords[:8]:
                add_terms(str(kw), 1.8 if " " in str(kw) else 1.2)

        if subject_tag:
            add_terms(subject_tag, 2.2)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
        return [term for term, _ in ranked[:max_terms]]

    def _context_terms_for_concept(self, concept_terms: list[str], material_context_terms: list[str]) -> list[str]:
        concept_tokens = normalize_terms(concept_terms)
        filtered = [term for term in material_context_terms if term not in concept_tokens]
        return filtered[:10]

    def _video_metadata_text(self, video: dict[str, Any]) -> str:
        pieces = [
            str(video.get("title") or "").strip(),
            str(video.get("description") or "").strip(),
            str(video.get("channel_title") or "").strip(),
        ]
        return " ".join(part for part in pieces if part).strip()

    def _score_text_relevance(
        self,
        conn,
        text: str,
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
    ) -> dict[str, Any]:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return {
                "score": -1.0,
                "embedding_sim": 0.0,
                "concept_overlap": 0.0,
                "context_overlap": 0.0,
                "subject_overlap": 0.0,
                "concept_hits": [],
                "context_hits": [],
                "matched_terms": [],
                "off_topic_penalty": 0.0,
                "passes": False,
            }

        concept_hits = self._extract_matched_terms(cleaned, concept_terms, limit=6)
        context_hits = self._extract_matched_terms(cleaned, context_terms, limit=6)
        concept_overlap = lexical_overlap_score(cleaned, concept_terms)
        context_overlap = lexical_overlap_score(cleaned, context_terms) if context_terms else 0.0
        subject_overlap = lexical_overlap_score(cleaned, [subject_tag]) if subject_tag else 0.0

        embedding_sim = 0.0
        if concept_embedding is not None:
            text_embedding = self.embedding_service.embed_texts(conn, [cleaned])[0]
            embedding_sim = float(text_embedding @ concept_embedding.astype(np.float32))

        allowed_terms = [*concept_terms, *context_terms]
        if subject_tag:
            allowed_terms.append(subject_tag)
        off_topic_penalty = self._off_topic_penalty(cleaned, allowed_terms=allowed_terms)

        score = (
            0.52 * max(0.0, embedding_sim)
            + 0.25 * concept_overlap
            + 0.17 * context_overlap
            + 0.08 * subject_overlap
            + 0.02 * min(4, len(concept_hits) + len(context_hits))
            - off_topic_penalty
        )
        score = float(max(-1.0, min(1.2, score)))

        merged_terms: list[str] = []
        for term in [*concept_hits, *context_hits]:
            if term not in merged_terms:
                merged_terms.append(term)

        return {
            "score": score,
            "embedding_sim": float(embedding_sim),
            "concept_overlap": float(concept_overlap),
            "context_overlap": float(context_overlap),
            "subject_overlap": float(subject_overlap),
            "concept_hits": concept_hits,
            "context_hits": context_hits,
            "matched_terms": merged_terms[:8],
            "off_topic_penalty": float(off_topic_penalty),
            "passes": False,
        }

    def _passes_relevance_gate(
        self,
        relevance: dict[str, Any],
        require_context: bool,
        fast_mode: bool,
    ) -> bool:
        concept_signal = float(relevance.get("concept_overlap") or 0.0) >= 0.05
        if not concept_signal:
            concept_signal = float(relevance.get("embedding_sim") or 0.0) >= (0.22 if fast_mode else 0.2)

        context_signal = True
        if require_context:
            context_signal = float(relevance.get("context_overlap") or 0.0) >= 0.04 or bool(relevance.get("context_hits"))

        off_topic_penalty = float(relevance.get("off_topic_penalty") or 0.0)
        if off_topic_penalty >= 0.24 and float(relevance.get("context_overlap") or 0.0) < 0.12:
            return False

        min_score = 0.1 if fast_mode else 0.08
        return concept_signal and context_signal and float(relevance.get("score") or -1.0) >= min_score

    def _rank_segments_by_relevance(
        self,
        conn,
        segments: list[SegmentMatch],
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        require_context: bool,
        fast_mode: bool,
    ) -> list[tuple[SegmentMatch, dict[str, Any]]]:
        ranked: list[tuple[SegmentMatch, dict[str, Any]]] = []
        for segment in segments:
            relevance = self._score_text_relevance(
                conn,
                text=segment.text,
                concept_terms=concept_terms,
                context_terms=context_terms,
                concept_embedding=concept_embedding,
                subject_tag=subject_tag,
            )
            passes = self._passes_relevance_gate(
                relevance=relevance,
                require_context=require_context,
                fast_mode=fast_mode,
            )
            relevance["passes"] = passes
            if not passes:
                continue

            combined_score = 0.65 * float(segment.score) + 0.35 * float(relevance["score"])
            ranked.append(
                (
                    SegmentMatch(
                        chunk_index=segment.chunk_index,
                        t_start=segment.t_start,
                        t_end=segment.t_end,
                        text=segment.text,
                        score=combined_score,
                    ),
                    relevance,
                )
            )

        ranked.sort(key=lambda row: row[0].score, reverse=True)
        return ranked

    def _merge_relevance_context(self, video_relevance: dict[str, Any], segment_relevance: dict[str, Any]) -> dict[str, Any]:
        merged_terms: list[str] = []
        for term in [*segment_relevance.get("matched_terms", []), *video_relevance.get("matched_terms", [])]:
            term_clean = str(term).strip()
            if not term_clean or term_clean in merged_terms:
                continue
            merged_terms.append(term_clean)

        context_hits: list[str] = []
        for term in [*segment_relevance.get("context_hits", []), *video_relevance.get("context_hits", [])]:
            term_clean = str(term).strip()
            if not term_clean or term_clean in context_hits:
                continue
            context_hits.append(term_clean)

        score = 0.44 * float(video_relevance.get("score") or 0.0) + 0.56 * float(segment_relevance.get("score") or 0.0)
        reason_parts: list[str] = []
        if merged_terms:
            reason_parts.append(f"Matched terms: {', '.join(merged_terms[:4])}")
        if context_hits:
            reason_parts.append(f"Material context: {', '.join(context_hits[:3])}")
        if not reason_parts:
            reason_parts.append("Matched semantically to the uploaded material")

        return {
            "score": float(score),
            "matched_terms": merged_terms[:8],
            "reason": ". ".join(reason_parts) + ".",
            "passes": bool(video_relevance.get("passes", True)) and bool(segment_relevance.get("passes", True)),
        }

    def _extract_matched_terms(self, text: str, terms: list[str], limit: int = 8) -> list[str]:
        text_lower = f" {text.lower()} "
        text_tokens = normalize_terms([text])
        hits: list[str] = []
        seen: set[str] = set()

        for raw_term in terms:
            term = " ".join(str(raw_term or "").lower().split()).strip()
            if not term:
                continue
            if " " in term and f" {term} " in text_lower:
                if term not in seen:
                    seen.add(term)
                    hits.append(term)
                if len(hits) >= limit:
                    break
                continue
            for token in normalize_terms([term]):
                if token in text_tokens and token not in seen:
                    seen.add(token)
                    hits.append(token)
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        return hits

    def _off_topic_penalty(self, text: str, allowed_terms: list[str]) -> float:
        lowered = " ".join(text.lower().split())
        allowed_tokens = normalize_terms(allowed_terms)
        text_tokens = normalize_terms([lowered])
        penalty = 0.0

        for phrase, weight in self.OFF_TOPIC_PHRASES.items():
            phrase_tokens = normalize_terms([phrase])
            if phrase in lowered and phrase_tokens.isdisjoint(allowed_tokens):
                penalty += weight

        for token in self.OFF_TOPIC_TOKENS:
            if token not in text_tokens:
                continue
            if token in allowed_tokens:
                continue
            if any(token.startswith(allowed) or allowed.startswith(token) for allowed in allowed_tokens):
                continue
            penalty += 0.035

        return min(0.42, penalty)

    def _clip_key(self, video_id: str, t_start: float, t_end: float) -> str:
        return f"{video_id}:{int(float(t_start))}:{int(float(t_end))}"

    def _order_concepts(self, conn, material_id: str, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        concept_counts = {
            row["concept_id"]: int(row["reel_count"])
            for row in fetch_all(
                conn,
                "SELECT concept_id, COUNT(*) AS reel_count FROM reels WHERE material_id = ? GROUP BY concept_id",
                (material_id,),
            )
        }
        concept_feedback = {
            row["concept_id"]: {
                "helpful": float(row["helpful_votes"]),
                "confusing": float(row["confusing_votes"]),
                "avg_rating": float(row["avg_rating"] or 3.0),
            }
            for row in fetch_all(
                conn,
                """
                SELECT
                    r.concept_id,
                    COALESCE(SUM(f.helpful), 0) AS helpful_votes,
                    COALESCE(SUM(f.confusing), 0) AS confusing_votes,
                    COALESCE(AVG(f.rating), 3.0) AS avg_rating
                FROM reels r
                LEFT JOIN reel_feedback f ON f.reel_id = r.id
                WHERE r.material_id = ?
                GROUP BY r.concept_id
                """,
                (material_id,),
            )
        }

        def concept_key(concept: dict[str, Any]) -> tuple[float, int, str]:
            feedback = concept_feedback.get(concept["id"], {"helpful": 0.0, "confusing": 0.0, "avg_rating": 3.0})
            mastery = self._concept_mastery(
                helpful=feedback["helpful"],
                confusing=feedback["confusing"],
                avg_rating=feedback["avg_rating"],
            )
            reel_count = concept_counts.get(concept["id"], 0)
            created = concept.get("created_at") or ""
            return (mastery, reel_count, created)

        return sorted(concepts, key=concept_key)

    def _concept_mastery(self, helpful: float, confusing: float, avg_rating: float) -> float:
        return 0.25 * helpful - 0.35 * confusing + 0.15 * (avg_rating - 3.0)

    def _upsert_video(self, conn, video: dict[str, Any]) -> None:
        upsert(
            conn,
            "videos",
            {
                "id": video["id"],
                "title": video["title"],
                "channel_title": video.get("channel_title", ""),
                "description": video.get("description", ""),
                "duration_sec": int(video.get("duration_sec") or 0),
                "is_creative_commons": 1 if video.get("is_creative_commons") else 0,
                "created_at": now_iso(),
            },
        )

    def _load_or_create_transcript_chunks(
        self,
        conn,
        video_id: str,
        transcript: list[dict[str, Any]],
    ) -> tuple[list[TranscriptChunk], np.ndarray]:
        rows = fetch_all(
            conn,
            "SELECT chunk_index, t_start, t_end, text, embedding_json FROM transcript_chunks WHERE video_id = ? ORDER BY chunk_index ASC",
            (video_id,),
        )
        if rows and all(row.get("embedding_json") for row in rows):
            chunks = [
                TranscriptChunk(
                    chunk_index=int(r["chunk_index"]),
                    t_start=float(r["t_start"]),
                    t_end=float(r["t_end"]),
                    text=r["text"],
                )
                for r in rows
            ]
            embeddings = np.array([json.loads(r["embedding_json"]) for r in rows], dtype=np.float32)
            return chunks, embeddings

        chunks = chunk_transcript(transcript)
        if not chunks:
            return [], np.empty((0, self.embedding_service.dim), dtype=np.float32)

        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(conn, texts)

        for chunk, emb in zip(chunks, embeddings):
            upsert(
                conn,
                "transcript_chunks",
                {
                    "id": str(uuid.uuid4()),
                    "video_id": video_id,
                    "chunk_index": chunk.chunk_index,
                    "t_start": chunk.t_start,
                    "t_end": chunk.t_end,
                    "text": chunk.text,
                    "embedding_json": dumps_json(emb.tolist()),
                    "created_at": now_iso(),
                },
            )

        return chunks, embeddings

    def _fast_segments_from_transcript(
        self,
        transcript: list[dict[str, Any]],
        concept_terms: list[str],
        max_segments: int = 2,
    ) -> list[SegmentMatch]:
        chunks = chunk_transcript(transcript)
        if not chunks:
            return self._fallback_segments_from_transcript(transcript)

        scored: list[SegmentMatch] = []
        for chunk in chunks:
            lexical = lexical_overlap_score(chunk.text, concept_terms)
            scored.append(
                SegmentMatch(
                    chunk_index=chunk.chunk_index,
                    t_start=chunk.t_start,
                    t_end=chunk.t_end,
                    text=chunk.text,
                    score=0.05 + 0.35 * lexical,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        selected: list[SegmentMatch] = []
        for candidate in scored:
            overlap = False
            for prev in selected:
                latest_start = max(candidate.t_start, prev.t_start)
                earliest_end = min(candidate.t_end, prev.t_end)
                if earliest_end - latest_start > 6:
                    overlap = True
                    break
            if overlap:
                continue
            selected.append(candidate)
            if len(selected) >= max_segments:
                break

        if not selected:
            return self._fallback_segments_from_transcript(transcript)
        return selected

    def _is_vague_concept(self, title: str, keywords: list[str], summary: str) -> bool:
        title_terms = normalize_terms([title])
        all_terms = normalize_terms([title, *keywords, summary])
        generic_terms = {
            "basics",
            "basic",
            "introduction",
            "intro",
            "overview",
            "beginner",
            "beginners",
            "tutorial",
            "guide",
            "fundamentals",
            "learn",
            "learning",
            "concept",
            "concepts",
        }
        if title_terms and title_terms.issubset(generic_terms):
            return True
        if len(title_terms) <= 1 and len(all_terms) <= 7:
            return True
        if len(summary.strip()) < 50 and len(keywords) < 3:
            return True
        return False

    def _split_video_into_short_segments(
        self,
        concept_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunks: list[TranscriptChunk],
        concept_terms: list[str],
        max_segments: int = 8,
    ) -> list[SegmentMatch]:
        if len(chunks) == 0 or len(chunk_embeddings) == 0:
            return []

        concept_vec = concept_embedding.astype(np.float32)
        sim_scores = (chunk_embeddings @ concept_vec).astype(np.float32)
        candidates: list[SegmentMatch] = []

        for idx, chunk in enumerate(chunks):
            text = chunk.text
            lexical = lexical_overlap_score(text, concept_terms)
            score = float(sim_scores[idx]) + 0.12 * lexical
            candidates.append(
                SegmentMatch(
                    chunk_index=chunk.chunk_index,
                    t_start=chunk.t_start,
                    t_end=chunk.t_end,
                    text=text,
                    score=score,
                )
            )

            if idx + 1 < len(chunks):
                nxt = chunks[idx + 1]
                start = chunk.t_start
                end = nxt.t_end
                length = end - start
                if 15 <= length <= 60:
                    pair_text = f"{chunk.text} {nxt.text}".strip()
                    pair_lexical = lexical_overlap_score(pair_text, concept_terms)
                    pair_score = float((sim_scores[idx] + sim_scores[idx + 1]) / 2.0) + 0.14 * pair_lexical + 0.02
                    candidates.append(
                        SegmentMatch(
                            chunk_index=chunk.chunk_index,
                            t_start=start,
                            t_end=end,
                            text=pair_text,
                            score=pair_score,
                        )
                    )

        candidates.sort(key=lambda item: item.score, reverse=True)

        selected: list[SegmentMatch] = []
        for cand in candidates:
            length = cand.t_end - cand.t_start
            if length < 15 or length > 60:
                continue
            overlap = False
            for prev in selected:
                latest_start = max(cand.t_start, prev.t_start)
                earliest_end = min(cand.t_end, prev.t_end)
                if earliest_end - latest_start > 6:
                    overlap = True
                    break
            if overlap:
                continue
            selected.append(cand)
            if len(selected) >= max_segments:
                break

        return selected

    def _merge_unique_segments(self, segments: list[SegmentMatch], max_items: int) -> list[SegmentMatch]:
        if not segments:
            return []
        deduped: list[SegmentMatch] = []
        for seg in sorted(segments, key=lambda item: item.score, reverse=True):
            is_dup = False
            for prev in deduped:
                if abs(seg.t_start - prev.t_start) <= 3 and abs(seg.t_end - prev.t_end) <= 3:
                    is_dup = True
                    break
            if is_dup:
                continue
            deduped.append(seg)
            if len(deduped) >= max_items:
                break
        return deduped

    def _fallback_segments_from_transcript(self, transcript: list[dict[str, Any]]) -> list[SegmentMatch]:
        if not transcript:
            return []

        entries = [entry for entry in transcript if (entry.get("text") or "").strip()]
        if not entries:
            return []

        start = float(entries[0].get("start") or 0.0)
        last = entries[-1]
        end = float(last.get("start") or start) + float(last.get("duration") or 0.0)
        if end <= start:
            return []

        snippet = " ".join(str(entry.get("text") or "").strip() for entry in entries[:40]).strip()
        if not snippet:
            return []

        return [
            SegmentMatch(
                chunk_index=0,
                t_start=start,
                t_end=end,
                text=snippet[:900],
                score=0.02,
            )
        ]

    def _fallback_segment_from_video_metadata(
        self,
        video: dict[str, Any],
        concept_terms: list[str],
    ) -> SegmentMatch | None:
        title = str(video.get("title") or "").strip()
        description = str(video.get("description") or "").strip()
        metadata_text = " ".join(part for part in [title, description] if part).strip()
        if not metadata_text:
            return None

        duration_sec = int(video.get("duration_sec") or 0)

        if duration_sec > 0:
            clip_len = min(45, max(15, duration_sec // 2 if duration_sec < 90 else 35))
            start = max(0.0, min(10.0, float(duration_sec - clip_len)))
            end = min(float(duration_sec), start + float(clip_len))
        else:
            start = 0.0
            end = 35.0

        if end - start < 15:
            end = start + 15.0

        lexical = lexical_overlap_score(metadata_text, concept_terms)
        return SegmentMatch(
            chunk_index=0,
            t_start=float(start),
            t_end=float(end),
            text=metadata_text[:900],
            score=0.04 + 0.08 * lexical,
        )

    def _create_reel(
        self,
        conn,
        material_id: str,
        concept: dict[str, Any],
        video: dict[str, Any],
        segment: SegmentMatch,
        clip_window: tuple[int, int] | None = None,
        transcript: list[dict[str, Any]] | None = None,
        relevance_context: dict[str, Any] | None = None,
        fast_mode: bool = False,
    ) -> dict[str, Any] | None:
        reel_id = str(uuid.uuid4())
        if clip_window is None:
            clip_window = self._normalize_clip_window(
                segment.t_start,
                segment.t_end,
                int(video.get("duration_sec") or 0),
            )
        if not clip_window:
            return None
        start_sec, end_sec = clip_window
        video_id = video["id"]
        url = (
            f"https://www.youtube.com/embed/{video_id}?start={start_sec}&end={end_sec}"
            f"&playlist={video_id}&autoplay=1&mute=1&playsinline=1"
            "&loop=1&controls=1&modestbranding=1&iv_load_policy=3&rel=0"
        )
        takeaways = build_takeaways(concept, segment.text)
        captions = self._build_caption_cues(
            transcript=transcript or [],
            clip_start=float(start_sec),
            clip_end=float(end_sec),
            fallback_text=segment.text,
        )
        video_title = str(video.get("title") or "").strip()
        video_description = self._clean_video_description(str(video.get("description") or ""))
        ai_summary = self._brief_ai_summary(
            conn,
            video_id=video_id,
            concept_title=str(concept.get("title") or ""),
            video_title=video_title,
            video_description=video_description,
            transcript_snippet=segment.text[:700],
            takeaways=takeaways,
            fast_mode=fast_mode,
        )
        relevance_score = float((relevance_context or {}).get("score") or segment.score)
        matched_terms = [
            str(term).strip()
            for term in (relevance_context or {}).get("matched_terms", [])
            if str(term).strip()
        ][:8]
        relevance_reason = str((relevance_context or {}).get("reason") or "").strip()

        try:
            upsert(
                conn,
                "reels",
                {
                    "id": reel_id,
                    "material_id": material_id,
                    "concept_id": concept["id"],
                    "video_id": video_id,
                    "video_url": url,
                    "t_start": float(start_sec),
                    "t_end": float(end_sec),
                    "transcript_snippet": segment.text[:700],
                    "takeaways_json": dumps_json(takeaways),
                    "base_score": float(segment.score),
                    "created_at": now_iso(),
                },
            )
        except sqlite3.IntegrityError:
            # DB-level uniqueness guard: skip duplicates safely if concurrent generation races.
            return None

        return {
            "reel_id": reel_id,
            "concept_id": concept["id"],
            "concept_title": concept["title"],
            "video_title": video_title,
            "video_description": video_description,
            "ai_summary": ai_summary,
            "video_url": url,
            "t_start": float(start_sec),
            "t_end": float(end_sec),
            "transcript_snippet": segment.text[:700],
            "takeaways": takeaways,
            "captions": captions,
            "score": float(segment.score),
            "relevance_score": relevance_score,
            "matched_terms": matched_terms,
            "relevance_reason": relevance_reason,
        }

    def _clean_video_description(self, description: str, max_chars: int = 7000) -> str:
        cleaned = description.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\r\n?", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned[:max_chars]

    def _fallback_ai_summary(
        self,
        *,
        concept_title: str,
        video_title: str,
        video_description: str,
        transcript_snippet: str,
        takeaways: list[str],
    ) -> str:
        takeaway_text = " ".join(t.strip() for t in takeaways if t.strip())
        candidates = [transcript_snippet.strip(), takeaway_text.strip(), video_description.strip()]
        source = next((c for c in candidates if c), "")
        if not source:
            return f"Brief overview of {concept_title or video_title or 'this reel'}."

        compact = " ".join(source.split())
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]
        if not sentences:
            summary = compact
        elif len(sentences) == 1:
            summary = sentences[0]
        else:
            summary = f"{sentences[0]} {sentences[1]}"
        summary = summary[:320].strip()
        if summary and summary[-1] not in ".!?":
            summary = f"{summary}."
        return summary

    def _brief_ai_summary(
        self,
        conn,
        *,
        video_id: str,
        concept_title: str,
        video_title: str,
        video_description: str,
        transcript_snippet: str,
        takeaways: list[str],
        fast_mode: bool = False,
    ) -> str:
        fallback = self._fallback_ai_summary(
            concept_title=concept_title,
            video_title=video_title,
            video_description=video_description,
            transcript_snippet=transcript_snippet,
            takeaways=takeaways,
        )
        if fast_mode:
            return fallback

        cache_payload = "|".join(
            [
                self.chat_model,
                video_id,
                concept_title[:120],
                video_title[:200],
                video_description[:1200],
                transcript_snippet[:1200],
                " ".join(takeaways[:4]),
            ]
        )
        cache_key = f"reel_ai_summary:{hashlib.sha256(cache_payload.encode('utf-8')).hexdigest()}"
        cached = fetch_one(conn, "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,))
        if cached:
            try:
                payload = json.loads(cached["response_json"])
                cached_summary = str(payload.get("summary") or "").strip()
                if cached_summary:
                    return cached_summary
            except (TypeError, json.JSONDecodeError):
                pass

        summary = fallback
        if self.openai_client:
            prompt = (
                "Write a brief study summary of this video clip in 1-2 sentences.\n"
                "Keep it concrete and under 220 characters.\n"
                "Do not add markdown or bullet points.\n\n"
                f"Concept: {concept_title or 'General topic'}\n"
                f"Video title: {video_title or 'Unknown'}\n"
                f"Video description: {video_description[:1500] or 'N/A'}\n"
                f"Clip transcript: {transcript_snippet[:1500] or 'N/A'}\n"
                f"Takeaways: {'; '.join(takeaways[:4]) or 'N/A'}"
            )
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.chat_model,
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": "You write concise educational summaries.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                generated = (response.choices[0].message.content or "").strip()
                if generated:
                    compact = " ".join(generated.split())
                    summary = compact[:320].strip() or fallback
            except Exception:
                summary = fallback

        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": cache_key,
                "response_json": dumps_json({"summary": summary}),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
        return summary

    def _normalize_clip_window(
        self,
        t_start: float,
        t_end: float,
        video_duration_sec: int,
        min_len: int = 15,
        max_len: int = 60,
        allow_exceed_max: bool = False,
        allow_below_min: bool = False,
    ) -> tuple[int, int] | None:
        if min_len < 1:
            min_len = 1

        start_sec = max(0, int(float(t_start)))
        end_sec = max(start_sec + 1, int(float(t_end)))

        if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
            end_sec = start_sec + max_len
        if not allow_below_min and end_sec - start_sec < min_len:
            end_sec = start_sec + min_len

        if video_duration_sec > 0:
            if not allow_below_min and video_duration_sec < min_len:
                return None
            if end_sec > video_duration_sec:
                end_sec = video_duration_sec
            if not allow_below_min and end_sec - start_sec < min_len:
                start_sec = max(0, end_sec - min_len)
            if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
                end_sec = start_sec + max_len
            if end_sec > video_duration_sec:
                end_sec = video_duration_sec

        if end_sec <= start_sec:
            return None
        if not allow_below_min and end_sec - start_sec < min_len:
            return None
        if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
            return None
        return (start_sec, end_sec)

    def _should_use_full_short_clip(self, prefer_short_query: bool, video_duration_sec: int) -> bool:
        if not prefer_short_query:
            return False
        if video_duration_sec <= 0:
            return False
        # Shorts are currently up to ~3 minutes; keep full playback for these.
        return video_duration_sec <= 185

    def _full_short_clip_window(self, video_duration_sec: int) -> tuple[int, int] | None:
        return self._normalize_clip_window(
            t_start=0.0,
            t_end=float(video_duration_sec),
            video_duration_sec=video_duration_sec,
            min_len=1,
            max_len=0,
            allow_exceed_max=True,
            allow_below_min=True,
        )

    def _refine_clip_window_from_transcript(
        self,
        transcript: list[dict[str, Any]],
        proposed_start: float,
        proposed_end: float,
        video_duration_sec: int,
    ) -> tuple[int, int] | None:
        entries: list[dict[str, Any]] = []
        for entry in transcript:
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            try:
                start = float(entry.get("start") or 0.0)
            except (TypeError, ValueError):
                continue
            duration_value = entry.get("duration")
            try:
                duration = float(duration_value) if duration_value is not None else 0.0
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0:
                duration = 1.5
            entries.append(
                {
                    "start": start,
                    "end": start + duration,
                    "text": text,
                }
            )

        if not entries:
            return self._normalize_clip_window(
                proposed_start,
                proposed_end,
                video_duration_sec,
                min_len=15,
                max_len=60,
            )

        desired_start = max(0.0, float(proposed_start))
        desired_end = max(desired_start + 1.0, float(proposed_end))

        start_idx = 0
        for i, item in enumerate(entries):
            if float(item["end"]) >= desired_start:
                start_idx = i
                break

        refined_start_idx = start_idx
        search_floor = max(0.0, desired_start - 6.0)
        for i in range(start_idx, -1, -1):
            if float(entries[i]["start"]) < search_floor:
                break
            if i == 0 or self._is_sentence_end(str(entries[i - 1]["text"])):
                refined_start_idx = i

        refined_start = float(entries[refined_start_idx]["start"])
        min_end = refined_start + 15.0
        max_end = refined_start + 60.0

        best_sentence_end: float | None = None
        best_sentence_cost = float("inf")
        best_any_end: float | None = None
        best_any_cost = float("inf")

        for i in range(refined_start_idx, len(entries)):
            item_end = float(entries[i]["end"])
            if item_end < min_end:
                continue
            if item_end > max_end + 2.0:
                break
            if item_end <= max_end:
                any_cost = abs(item_end - desired_end)
                if any_cost < best_any_cost:
                    best_any_cost = any_cost
                    best_any_end = item_end
                if self._is_sentence_end(str(entries[i]["text"])):
                    sent_cost = abs(item_end - desired_end)
                    if sent_cost < best_sentence_cost:
                        best_sentence_cost = sent_cost
                        best_sentence_end = item_end

        refined_end = best_sentence_end if best_sentence_end is not None else best_any_end
        if refined_end is None:
            refined_end = min(max_end, max(min_end, desired_end))

        return self._normalize_clip_window(
            refined_start,
            refined_end,
            video_duration_sec,
            min_len=15,
            max_len=60,
        )

    def _is_sentence_end(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        return bool(re.search(r"[.!?…][\"'\)\]]*$", cleaned))

    def _build_caption_cues(
        self,
        transcript: list[dict[str, Any]],
        clip_start: float,
        clip_end: float,
        fallback_text: str | None = None,
    ) -> list[dict[str, Any]]:
        if clip_end <= clip_start:
            return []

        clip_len = max(0.2, float(clip_end - clip_start))
        cues: list[dict[str, Any]] = []

        for entry in transcript:
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue

            try:
                entry_start = float(entry.get("start") or 0.0)
            except (TypeError, ValueError):
                continue

            duration_value = entry.get("duration")
            try:
                entry_duration = float(duration_value) if duration_value is not None else 0.0
            except (TypeError, ValueError):
                entry_duration = 0.0

            if entry_duration <= 0:
                entry_duration = 1.8

            entry_end = entry_start + entry_duration
            if entry_end <= clip_start or entry_start >= clip_end:
                continue

            cue_start = max(0.0, max(entry_start, clip_start) - clip_start)
            cue_end = min(clip_len, min(entry_end, clip_end) - clip_start)
            if cue_end - cue_start < 0.16:
                cue_end = min(clip_len, cue_start + 0.9)

            payload = {
                "start": round(float(cue_start), 2),
                "end": round(float(min(clip_len, max(cue_end, cue_start + 0.16))), 2),
                "text": text[:220],
            }

            if cues and payload["text"] == cues[-1]["text"] and payload["start"] - cues[-1]["end"] <= 0.2:
                cues[-1]["end"] = payload["end"]
            else:
                cues.append(payload)

            if len(cues) >= 140:
                break

        if not cues and fallback_text and fallback_text.strip():
            cues.append(
                {
                    "start": 0.0,
                    "end": round(clip_len, 2),
                    "text": fallback_text.strip()[:240],
                }
            )
        return cues

    def record_feedback(
        self,
        conn,
        reel_id: str,
        helpful: bool,
        confusing: bool,
        rating: int | None,
        saved: bool,
    ) -> None:
        upsert(
            conn,
            "reel_feedback",
            {
                "id": str(uuid.uuid4()),
                "reel_id": reel_id,
                "helpful": 1 if helpful else 0,
                "confusing": 1 if confusing else 0,
                "rating": rating,
                "saved": 1 if saved else 0,
                "created_at": now_iso(),
            },
        )

    def ranked_feed(self, conn, material_id: str, fast_mode: bool = False) -> list[dict[str, Any]]:
        material = fetch_one(conn, "SELECT subject_tag FROM materials WHERE id = ?", (material_id,))
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None

        reel_rows = fetch_all(
            conn,
            """
            SELECT
                r.id AS reel_id,
                r.concept_id,
                r.video_id,
                c.title AS concept_title,
                c.keywords_json AS concept_keywords_json,
                c.summary AS concept_summary,
                c.embedding_json AS concept_embedding_json,
                v.title AS video_title,
                COALESCE(v.description, '') AS video_description,
                r.video_url,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                r.takeaways_json,
                r.base_score,
                COALESCE(SUM(f.helpful), 0) AS helpful_votes,
                COALESCE(SUM(f.confusing), 0) AS confusing_votes,
                COALESCE(AVG(f.rating), 3.0) AS avg_rating,
                COALESCE(SUM(f.saved), 0) AS saves,
                r.created_at
            FROM reels r
            JOIN concepts c ON c.id = r.concept_id
            JOIN videos v ON v.id = r.video_id
            LEFT JOIN reel_feedback f ON f.reel_id = r.id
            WHERE r.material_id = ?
              AND (r.t_end - r.t_start) >= 1
            GROUP BY r.id, c.title, c.keywords_json, c.summary, c.embedding_json, v.title, v.description
            """,
            (material_id,),
        )

        video_ids = sorted({str(row["video_id"]) for row in reel_rows if row.get("video_id")})
        transcript_by_video: dict[str, list[dict[str, Any]]] = {}
        if video_ids:
            placeholders = ", ".join(["?"] * len(video_ids))
            transcript_rows = fetch_all(
                conn,
                f"SELECT video_id, transcript_json FROM transcript_cache WHERE video_id IN ({placeholders})",
                tuple(video_ids),
            )
            for trow in transcript_rows:
                try:
                    transcript_by_video[str(trow["video_id"])] = json.loads(trow["transcript_json"])
                except (TypeError, json.JSONDecodeError):
                    transcript_by_video[str(trow["video_id"])] = []

        concept_feedback = fetch_all(
            conn,
            """
            SELECT
                r.concept_id,
                COALESCE(SUM(f.helpful), 0) AS helpful_votes,
                COALESCE(SUM(f.confusing), 0) AS confusing_votes
            FROM reels r
            LEFT JOIN reel_feedback f ON f.reel_id = r.id
            WHERE r.material_id = ?
            GROUP BY r.concept_id
            """,
            (material_id,),
        )
        concept_signal = {
            row["concept_id"]: (float(row["helpful_votes"]), float(row["confusing_votes"]))
            for row in concept_feedback
        }

        concept_rows = fetch_all(
            conn,
            "SELECT id, title, keywords_json, summary, embedding_json FROM concepts WHERE material_id = ? ORDER BY created_at ASC, title ASC",
            (material_id,),
        )
        concept_order = {row["id"]: idx + 1 for idx, row in enumerate(concept_rows)}
        concept_by_id = {row["id"]: row for row in concept_rows}
        total_concepts = len(concept_rows)
        material_context_terms = self._build_material_context_terms(concepts=concept_rows, subject_tag=subject_tag)

        scored: list[dict[str, Any]] = []
        for row in reel_rows:
            concept_helpful, concept_confusing = concept_signal.get(row["concept_id"], (0.0, 0.0))
            try:
                takeaways = json.loads(row["takeaways_json"])
            except (TypeError, json.JSONDecodeError):
                takeaways = []
            if not isinstance(takeaways, list):
                takeaways = []
            video_title = str(row.get("video_title") or "").strip()
            video_description = self._clean_video_description(str(row.get("video_description") or ""))
            transcript_snippet = str(row.get("transcript_snippet") or "")
            concept_row = concept_by_id.get(row["concept_id"], row)
            try:
                concept_keywords = json.loads(str(concept_row.get("concept_keywords_json") or concept_row.get("keywords_json") or "[]"))
            except json.JSONDecodeError:
                concept_keywords = []
            if not isinstance(concept_keywords, list):
                concept_keywords = []
            concept_title = str(concept_row.get("concept_title") or concept_row.get("title") or row.get("concept_title") or "").strip()
            concept_summary = str(concept_row.get("concept_summary") or concept_row.get("summary") or "").strip()
            concept_terms = [concept_title, *[str(k) for k in concept_keywords[:8]], concept_summary]
            context_terms = self._context_terms_for_concept(concept_terms, material_context_terms)

            concept_embedding: np.ndarray | None = None
            if not fast_mode:
                embedding_json = str(
                    concept_row.get("concept_embedding_json")
                    or concept_row.get("embedding_json")
                    or row.get("concept_embedding_json")
                    or ""
                )
                if embedding_json:
                    try:
                        concept_embedding = np.array(json.loads(embedding_json), dtype=np.float32)
                    except (TypeError, json.JSONDecodeError):
                        concept_embedding = None

            relevance = self._score_text_relevance(
                conn,
                text=" ".join([video_title, video_description, transcript_snippet]).strip(),
                concept_terms=concept_terms,
                context_terms=context_terms,
                concept_embedding=concept_embedding,
                subject_tag=subject_tag,
            )
            relevance["passes"] = self._passes_relevance_gate(
                relevance=relevance,
                require_context=bool(context_terms),
                fast_mode=fast_mode,
            )
            if not relevance["passes"] and (
                float(relevance.get("off_topic_penalty") or 0.0) >= 0.24
                or float(relevance.get("score") or -1.0) < 0.02
            ):
                # Hide strongly off-topic clips that can still exist from older generations.
                continue
            relevance_context = self._merge_relevance_context(relevance, relevance)

            ai_summary = self._brief_ai_summary(
                conn,
                video_id=str(row.get("video_id") or ""),
                concept_title=concept_title,
                video_title=video_title,
                video_description=video_description,
                transcript_snippet=transcript_snippet,
                takeaways=takeaways,
                fast_mode=fast_mode,
            )
            score = (
                float(row["base_score"])
                + 0.18 * float(row["helpful_votes"])
                - 0.22 * float(row["confusing_votes"])
                + 0.06 * (float(row["avg_rating"] or 3.0) - 3.0)
                + 0.05 * float(row["saves"])
                + 0.04 * concept_helpful
                - 0.06 * concept_confusing
                + 0.22 * float(relevance_context.get("score") or 0.0)
                - 0.12 * float(relevance.get("off_topic_penalty") or 0.0)
            )
            scored.append(
                {
                    "reel_id": row["reel_id"],
                    "video_id": row["video_id"],
                    "concept_id": row["concept_id"],
                    "concept_title": concept_title,
                    "video_title": video_title,
                    "video_description": video_description,
                    "ai_summary": ai_summary,
                    "video_url": row["video_url"],
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "transcript_snippet": transcript_snippet,
                    "takeaways": takeaways,
                    "captions": self._build_caption_cues(
                        transcript=transcript_by_video.get(str(row.get("video_id")), []),
                        clip_start=float(row["t_start"]),
                        clip_end=float(row["t_end"]),
                        fallback_text=transcript_snippet,
                    ),
                    "score": score,
                    "relevance_score": float(relevance_context.get("score") or 0.0),
                    "matched_terms": relevance_context.get("matched_terms", []),
                    "relevance_reason": str(relevance_context.get("reason") or ""),
                    "concept_position": concept_order.get(row["concept_id"]),
                    "total_concepts": total_concepts,
                    "created_at": row["created_at"],
                }
            )

        scored.sort(key=lambda x: (x["score"], x["created_at"]), reverse=True)
        deduped: list[dict[str, Any]] = []
        seen_reel_ids: set[str] = set()
        seen_clip_keys: set[str] = set()
        for item in scored:
            reel_id = str(item.get("reel_id") or "")
            if reel_id and reel_id in seen_reel_ids:
                continue
            video_id = str(item.get("video_id") or "")
            if not video_id:
                continue
            clip_key = self._clip_key(video_id, float(item.get("t_start") or 0), float(item.get("t_end") or 0))
            if clip_key in seen_clip_keys:
                continue
            if reel_id:
                seen_reel_ids.add(reel_id)
            seen_clip_keys.add(clip_key)
            clean_item = dict(item)
            clean_item.pop("video_id", None)
            deduped.append(clean_item)
        return deduped
