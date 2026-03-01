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

        existing_video_ids = {
            str(r["video_id"])
            for r in fetch_all(
                conn,
                "SELECT video_id FROM reels WHERE material_id = ?",
                (material_id,),
            )
            if r.get("video_id")
        }

        generated: list[dict[str, Any]] = []
        material = fetch_one(conn, "SELECT subject_tag FROM materials WHERE id = ?", (material_id,))
        subject_tag = material["subject_tag"] if material else None

        for concept in concepts:
            concept_keywords = json.loads(concept["keywords_json"])
            concept_embedding = self._get_concept_embedding(conn, concept) if not fast_mode else None
            concept_terms = [concept["title"], *concept_keywords, concept.get("summary", "")]
            vague_topic = self._is_vague_concept(
                title=concept["title"],
                keywords=concept_keywords,
                summary=concept.get("summary", ""),
            )
            queries = self._build_query_variants(concept["title"], concept_keywords, subject_tag)
            if fast_mode:
                queries = queries[:2]
            seen_video_ids: set[str] = set()

            for query in queries:
                # First pull short-form videos; then broaden to any duration for long-form clipping.
                durations = ("short", None) if not fast_mode else ("short",)
                for duration in durations:
                    prefer_short_query = duration == "short"
                    videos = self.youtube_service.search_videos(
                        conn,
                        query=query,
                        max_results=4 if fast_mode else 8,
                        creative_commons_only=creative_commons_only,
                        video_duration=duration,
                    )

                    for video in videos:
                        if video["id"] in seen_video_ids:
                            continue
                        seen_video_ids.add(video["id"])
                        if video["id"] in existing_video_ids:
                            continue
                        video_duration = int(video.get("duration_sec") or 0)

                        self._upsert_video(conn, video)
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

                        for segment in segments:
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
                            reel = self._create_reel(
                                conn,
                                material_id=material_id,
                                concept=concept,
                                video=video,
                                segment=segment,
                                clip_window=clip_window,
                                transcript=transcript,
                                fast_mode=fast_mode,
                            )
                            if not reel:
                                continue
                            generated.append(reel)
                            existing_video_ids.add(video["id"])

                            if len(generated) >= num_reels:
                                return generated
                            # Keep only one reel per video to avoid repeated content.
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

    def _build_query_variants(self, title: str, keywords: list[str], subject_tag: str | None) -> list[str]:
        short_keywords = keywords[:3]
        subject = f"{subject_tag} " if subject_tag else ""
        variants = [
            f"{subject}{title} {' '.join(short_keywords)} tutorial explanation",
            f"{subject}{title} {' '.join(short_keywords[:2])} for beginners",
            f"{subject}{title} shorts explained",
            f"{subject}{title} practice problems solved",
            f"{subject}{title} crash course",
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
        reel_rows = fetch_all(
            conn,
            """
            SELECT
                r.id AS reel_id,
                r.concept_id,
                r.video_id,
                c.title AS concept_title,
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
            GROUP BY r.id, c.title, v.title, v.description
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
            "SELECT id FROM concepts WHERE material_id = ? ORDER BY created_at ASC, title ASC",
            (material_id,),
        )
        concept_order = {row["id"]: idx + 1 for idx, row in enumerate(concept_rows)}
        total_concepts = len(concept_rows)

        scored: list[dict[str, Any]] = []
        for row in reel_rows:
            concept_helpful, concept_confusing = concept_signal.get(row["concept_id"], (0.0, 0.0))
            takeaways = json.loads(row["takeaways_json"])
            video_title = str(row.get("video_title") or "").strip()
            video_description = self._clean_video_description(str(row.get("video_description") or ""))
            ai_summary = self._brief_ai_summary(
                conn,
                video_id=str(row.get("video_id") or ""),
                concept_title=str(row.get("concept_title") or ""),
                video_title=video_title,
                video_description=video_description,
                transcript_snippet=str(row.get("transcript_snippet") or ""),
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
            )
            scored.append(
                {
                    "reel_id": row["reel_id"],
                    "video_id": row["video_id"],
                    "concept_id": row["concept_id"],
                    "concept_title": row["concept_title"],
                    "video_title": video_title,
                    "video_description": video_description,
                    "ai_summary": ai_summary,
                    "video_url": row["video_url"],
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "transcript_snippet": row["transcript_snippet"],
                    "takeaways": takeaways,
                    "captions": self._build_caption_cues(
                        transcript=transcript_by_video.get(str(row.get("video_id")), []),
                        clip_start=float(row["t_start"]),
                        clip_end=float(row["t_end"]),
                        fallback_text=str(row.get("transcript_snippet") or ""),
                    ),
                    "score": score,
                    "concept_position": concept_order.get(row["concept_id"]),
                    "total_concepts": total_concepts,
                    "created_at": row["created_at"],
                }
            )

        scored.sort(key=lambda x: (x["score"], x["created_at"]), reverse=True)
        deduped: list[dict[str, Any]] = []
        seen_reel_ids: set[str] = set()
        seen_video_ids: set[str] = set()
        seen_clip_keys: set[str] = set()
        for item in scored:
            reel_id = str(item.get("reel_id") or "")
            if reel_id and reel_id in seen_reel_ids:
                continue
            video_id = str(item.get("video_id") or "")
            clip_key = f"{video_id}:{int(float(item.get('t_start') or 0))}:{int(float(item.get('t_end') or 0))}"
            if clip_key in seen_clip_keys:
                continue
            if not video_id or video_id in seen_video_ids:
                continue
            if reel_id:
                seen_reel_ids.add(reel_id)
            seen_video_ids.add(video_id)
            seen_clip_keys.add(clip_key)
            clean_item = dict(item)
            clean_item.pop("video_id", None)
            deduped.append(clean_item)
        return deduped
