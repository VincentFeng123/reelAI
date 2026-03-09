import hashlib
import json
import os
import re
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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


@dataclass
class QueryCandidate:
    text: str
    strategy: str
    confidence: float
    source_terms: list[str] = field(default_factory=list)
    weight: float = 1.0
    stage: str = "broad"
    source_surface: str = "youtube_api"


@dataclass
class RetrievalStagePlan:
    name: str
    queries: list[QueryCandidate]
    budget: int
    min_good_results: int


class ReelService:
    VALID_VIDEO_POOL_MODES = {"short-first", "balanced", "long-form"}
    VALID_VIDEO_DURATION_PREFS = {"any", "short", "medium", "long"}
    DEFAULT_TARGET_CLIP_DURATION_SEC = 55
    MIN_TARGET_CLIP_DURATION_SEC = 15
    MAX_TARGET_CLIP_DURATION_SEC = 180
    MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC = 15
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
    QUERY_STRATEGY_PRIOR: dict[str, float] = {
        "literal": 0.78,
        "paraphrase": 0.72,
        "scene": 0.84,
        "object": 0.8,
        "action": 0.79,
        "broll": 0.87,
        "news_doc": 0.73,
        "tutorial_demo": 0.75,
        "recovery_adjacent": 0.64,
    }
    SOURCE_SURFACE_PRIOR: dict[str, float] = {
        "youtube_api": 1.0,
        "youtube_html": 0.94,
        "duckduckgo_site": 0.87,
        "bing_site": 0.85,
        "duckduckgo_quoted": 0.9,
        "bing_quoted": 0.88,
        "local_cache": 0.78,
    }
    CHANNEL_QUALITY_BONUS: dict[str, float] = {
        "news": 0.06,
        "education": 0.06,
        "tutorial": 0.04,
        "stock_footage": 0.08,
        "podcast": -0.03,
        "low_quality_compilation": -0.11,
    }
    CLIPABILITY_PENALTY_TOKENS = {
        "live stream",
        "podcast",
        "full episode",
        "compilation",
        "reaction",
    }
    QUERY_RETRIEVAL_WORKERS_FAST = 6
    QUERY_RETRIEVAL_WORKERS_SLOW = 6
    TRANSCRIPT_FETCH_WORKERS_FAST = 6
    TRANSCRIPT_FETCH_WORKERS_SLOW = 6

    def __init__(self, embedding_service, youtube_service) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service
        self.youtube_service = youtube_service
        self.chat_model = settings.openai_chat_model
        self.retrieval_engine_v2_enabled = bool(settings.retrieval_engine_v2_enabled)
        self.retrieval_tier2_enabled = bool(settings.retrieval_tier2_enabled)
        self.retrieval_debug_logging = bool(settings.retrieval_debug_logging)
        self.serverless_mode = bool(
            os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE")
        )
        self._strategy_history_cache: dict[str, float] = {}
        allow_openai_serverless = os.getenv("ALLOW_OPENAI_IN_SERVERLESS") == "1"
        can_use_openai = (
            bool(settings.openai_enabled)
            and bool(settings.openai_api_key)
            and (not self.serverless_mode or allow_openai_serverless)
        )
        self.openai_client = OpenAI(api_key=settings.openai_api_key, timeout=8.0) if can_use_openai else None

    def generate_reels(
        self,
        conn,
        material_id: str,
        concept_id: str | None,
        num_reels: int,
        creative_commons_only: bool,
        fast_mode: bool = False,
        video_pool_mode: str = "short-first",
        preferred_video_duration: str = "any",
        target_clip_duration_sec: int = DEFAULT_TARGET_CLIP_DURATION_SEC,
        target_clip_duration_min_sec: int | None = None,
        target_clip_duration_max_sec: int | None = None,
        dry_run: bool = False,
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
        if self.serverless_mode:
            concepts = concepts[:1]
        elif fast_mode:
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
        safe_video_pool_mode = self._normalize_video_pool_mode(video_pool_mode)
        safe_video_duration_pref = self._normalize_preferred_video_duration(preferred_video_duration)
        max_generation_target = self._generation_target_cap(
            num_reels=num_reels,
            preferred_video_duration=safe_video_duration_pref,
            fast_mode=fast_mode,
        )
        if self.serverless_mode:
            max_generation_target = min(max_generation_target, max(3, num_reels))
        clip_min_len, clip_max_len, safe_target_clip_duration = self._resolve_clip_duration_bounds(
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )

        generated: list[dict[str, Any]] = []
        material = fetch_one(conn, "SELECT subject_tag FROM materials WHERE id = ?", (material_id,))
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None
        material_context_terms = self._build_material_context_terms(concepts=concepts, subject_tag=subject_tag)
        for concept in concepts:
            if self._should_finalize_generation(
                generated=generated,
                num_reels=num_reels,
                preferred_video_duration=safe_video_duration_pref,
                max_generation_target=max_generation_target,
            ):
                break

            concept_keywords = self._parse_keywords_json(concept.get("keywords_json"))
            concept_summary = str(concept.get("summary") or "")
            concept_embedding: np.ndarray | None = None
            if not fast_mode:
                try:
                    concept_embedding = self._get_concept_embedding(conn, concept)
                except Exception:
                    concept_embedding = None

            concept_terms = [concept["title"], *concept_keywords, concept_summary]
            context_terms = self._context_terms_for_concept(concept_terms, material_context_terms)
            vague_topic = self._is_vague_concept(
                title=concept["title"],
                keywords=concept_keywords,
                summary=concept_summary,
            )
            visual_spec = self._derive_visual_scene_spec(
                title=concept["title"],
                keywords=concept_keywords,
                summary=concept_summary,
            )
            if self.retrieval_engine_v2_enabled:
                query_candidates = self._build_query_candidates(
                    title=concept["title"],
                    keywords=concept_keywords,
                    summary=concept_summary,
                    subject_tag=subject_tag,
                    context_terms=context_terms,
                    visual_spec=visual_spec,
                    fast_mode=fast_mode,
                )
            else:
                query_candidates = [
                    QueryCandidate(
                        text=q,
                        strategy="literal",
                        confidence=0.7,
                        source_terms=[concept["title"]],
                        stage="broad",
                    )
                    for q in self._build_query_variants(
                        concept["title"],
                        concept_keywords,
                        subject_tag,
                        context_terms=context_terms,
                    )
                ]
            retrieval_stages = self._build_retrieval_stage_plan(query_candidates=query_candidates, fast_mode=fast_mode)
            retrieval_run = self._init_retrieval_debug_run(
                material_id=material_id,
                concept_id=str(concept.get("id") or ""),
                concept_title=str(concept.get("title") or ""),
            )

            seen_video_ids: set[str] = set()
            stage_candidates: list[dict[str, Any]] = []
            all_query_reports: list[dict[str, Any]] = []

            primary_stages = [stage for stage in retrieval_stages if stage.name == "high_precision"]
            expansion_stages = [stage for stage in retrieval_stages if stage.name != "high_precision"]
            pass_groups = [primary_stages, expansion_stages]

            for pass_index, stage_group in enumerate(pass_groups):
                if not stage_group:
                    continue
                if pass_index > 0 and self._fast_pass_is_sufficient(
                    stage_candidates,
                    fast_mode=fast_mode,
                    max_generation_target=max_generation_target,
                ):
                    break

                for stage in stage_group:
                    if self._should_finalize_generation(
                        generated=generated,
                        num_reels=num_reels,
                        preferred_video_duration=safe_video_duration_pref,
                        max_generation_target=max_generation_target,
                    ):
                        break

                    good_results = 0
                    stage_duration_plan = self._stage_duration_plan(
                        stage_name=stage.name,
                        preferred_video_duration=safe_video_duration_pref,
                        video_pool_mode=safe_video_pool_mode,
                        fast_mode=fast_mode,
                    )
                    strict_duration = stage.name == "high_precision" and safe_video_duration_pref in {
                        "short",
                        "medium",
                        "long",
                    }
                    allow_unknown_duration = stage.name != "high_precision"
                    stage_queries = stage.queries[: stage.budget]
                    stage_query_reports = [
                        {
                            "query": query_candidate.text,
                            "strategy": query_candidate.strategy,
                            "stage": stage.name,
                            "source_terms": query_candidate.source_terms,
                            "weight": float(query_candidate.weight),
                            "surface": query_candidate.source_surface,
                            "results": 0,
                            "kept": 0,
                        }
                        for query_candidate in stage_queries
                    ]
                    max_results_for_query = self._search_results_budget(
                        fast_mode=fast_mode,
                        generated_count=len(generated),
                        max_generation_target=max_generation_target,
                    )
                    search_jobs = self._stage_search_jobs_parallel(
                        stage_name=stage.name,
                        stage_queries=stage_queries,
                        stage_duration_plan=stage_duration_plan,
                        max_results_for_query=max_results_for_query,
                        creative_commons_only=creative_commons_only,
                        fast_mode=fast_mode,
                    )

                    for query_idx, _duration_idx, query_candidate, _duration, videos in search_jobs:
                        if query_idx >= len(stage_query_reports):
                            continue
                        query_report = stage_query_reports[query_idx]
                        query_report["results"] += len(videos)

                        for video in videos:
                            video_id = str(video.get("id") or "").strip()
                            if not video_id or video_id in seen_video_ids:
                                continue
                            seen_video_ids.add(video_id)
                            existing_for_video = existing_video_counts.get(video_id, 0)
                            generated_for_video = generated_video_counts.get(video_id, 0)
                            if existing_for_video + generated_for_video >= max_segments_per_video:
                                continue

                            video_duration = int(video.get("duration_sec") or 0)
                            if strict_duration and not self._video_matches_preferred_duration(
                                video_duration_sec=video_duration,
                                preferred_video_duration=safe_video_duration_pref,
                                allow_unknown_duration=allow_unknown_duration,
                            ):
                                continue

                            if not dry_run:
                                self._upsert_video(conn, video)

                            ranking = self._score_video_candidate(
                                conn,
                                video=video,
                                query_candidate=query_candidate,
                                concept_terms=concept_terms,
                                context_terms=context_terms,
                                concept_embedding=concept_embedding,
                                subject_tag=subject_tag,
                                visual_spec=visual_spec,
                                preferred_video_duration=safe_video_duration_pref,
                                stage_name=stage.name,
                                require_context=bool(context_terms) and vague_topic,
                                fast_mode=fast_mode,
                            )
                            if not bool(ranking.get("passes", False)):
                                continue
                            query_report["kept"] += 1
                            if float(ranking.get("discovery_score") or 0.0) >= 0.2:
                                good_results += 1

                            stage_candidates.append(
                                {
                                    "video": video,
                                    "video_id": video_id,
                                    "video_duration": video_duration,
                                    "video_relevance": ranking["text_relevance"],
                                    "ranking": ranking,
                                    "query_candidate": query_candidate,
                                    "stage": stage.name,
                                }
                            )

                    all_query_reports.extend(stage_query_reports)
                    if good_results >= stage.min_good_results and len(stage_candidates) >= stage.min_good_results:
                        break

                if pass_index == 0 and self._fast_pass_is_sufficient(
                    stage_candidates,
                    fast_mode=fast_mode,
                    max_generation_target=max_generation_target,
                ):
                    break

            if not stage_candidates:
                local_candidates = self._recover_candidates_from_local_corpus(
                    conn,
                    concept_terms=concept_terms,
                    context_terms=context_terms,
                    concept_embedding=concept_embedding,
                    subject_tag=subject_tag,
                    visual_spec=visual_spec,
                    preferred_video_duration=safe_video_duration_pref,
                    fast_mode=fast_mode,
                    existing_video_counts=existing_video_counts,
                    generated_video_counts=generated_video_counts,
                    max_segments_per_video=max_segments_per_video,
                    concept_title=str(concept.get("title") or ""),
                )
                if local_candidates:
                    stage_candidates = local_candidates
                    all_query_reports.append(
                        {
                            "query": f"local_cache:{concept.get('title')}",
                            "strategy": "recovery_adjacent",
                            "stage": "recovery",
                            "source_terms": concept_terms[:4],
                            "weight": 0.5,
                            "surface": "local_cache",
                            "results": len(local_candidates),
                            "kept": len(local_candidates),
                        }
                    )

            if not stage_candidates:
                self._persist_retrieval_debug_run(
                    conn,
                    run=retrieval_run,
                    query_reports=all_query_reports,
                    candidate_rows=[],
                    selected=None,
                    failure_reason="no_candidates_after_retrieval",
                    dry_run=dry_run,
                )
                continue

            if self.retrieval_tier2_enabled:
                stage_candidates = self._collapse_near_duplicate_candidates(stage_candidates)
            else:
                stage_candidates = sorted(
                    stage_candidates,
                    key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0),
                    reverse=True,
                )
            transcript_budget = min(
                20,
                self._transcript_expansion_budget(
                    fast_mode=fast_mode,
                    generated_count=len(generated),
                    max_generation_target=max_generation_target,
                ),
            )
            if self.retrieval_tier2_enabled:
                ranked_candidates = self._diversify_video_candidates(
                    stage_candidates,
                    top_k=max(12, transcript_budget * 2),
                )
            else:
                ranked_candidates = stage_candidates[: max(12, transcript_budget * 2)]

            selected_outcome: dict[str, Any] | None = None
            candidate_records: list[dict[str, Any]] = []
            for candidate in ranked_candidates[: max(20, transcript_budget * 2)]:
                ranking = dict(candidate.get("ranking") or {})
                candidate_records.append(
                    {
                        "video_id": str(candidate.get("video_id") or ""),
                        "video_title": str((candidate.get("video") or {}).get("title") or ""),
                        "channel_title": str((candidate.get("video") or {}).get("channel_title") or ""),
                        "strategy": str((candidate.get("query_candidate") or QueryCandidate("", "", 0.0)).strategy or ""),
                        "stage": str(candidate.get("stage") or ""),
                        "query": str((candidate.get("query_candidate") or QueryCandidate("", "", 0.0)).text or ""),
                        "final_score": float(ranking.get("final_score") or 0.0),
                        "discovery_score": float(ranking.get("discovery_score") or 0.0),
                        "clipability_score": float(ranking.get("clipability_score") or 0.0),
                        "source_surface": str((candidate.get("video") or {}).get("search_source") or ""),
                        "features": ranking.get("features") or {},
                    }
                )

            transcript_candidates = ranked_candidates[:transcript_budget]
            transcript_prefetch_ids: list[str] = []
            for candidate in transcript_candidates:
                video_id = str(candidate.get("video_id") or "")
                video_duration = int(candidate.get("video_duration") or 0)
                use_full_short_clip = self._should_use_full_short_clip(
                    prefer_short_query=self._video_duration_bucket(video_duration) == "short",
                    video_duration_sec=video_duration,
                    clip_min_len=clip_min_len,
                    clip_max_len=clip_max_len,
                )
                if not use_full_short_clip:
                    transcript_prefetch_ids.append(video_id)
            transcript_cache = self._prefetch_transcripts_parallel(
                transcript_prefetch_ids,
                fast_mode=fast_mode,
            )

            for candidate in transcript_candidates:
                video = candidate["video"]
                video_id = str(candidate["video_id"])
                video_duration = int(candidate["video_duration"])
                ranking = dict(candidate.get("ranking") or {})
                query_candidate = candidate.get("query_candidate")
                if not isinstance(query_candidate, QueryCandidate):
                    query_candidate = QueryCandidate(text="", strategy="literal", confidence=0.5)

                existing_for_video = existing_video_counts.get(video_id, 0)
                generated_for_video = generated_video_counts.get(video_id, 0)
                if existing_for_video + generated_for_video >= max_segments_per_video:
                    continue

                use_full_short_clip = self._should_use_full_short_clip(
                    prefer_short_query=self._video_duration_bucket(video_duration) == "short",
                    video_duration_sec=video_duration,
                    clip_min_len=clip_min_len,
                    clip_max_len=clip_max_len,
                )
                transcript = [] if use_full_short_clip else list(transcript_cache.get(video_id) or [])
                if transcript:
                    transcript_ranking = self._score_transcript_alignment(
                        conn,
                        transcript=transcript,
                        concept_terms=concept_terms,
                        concept_embedding=concept_embedding,
                        visual_spec=visual_spec,
                    )
                    ranking["discovery_score"] = min(
                        1.0,
                        float(ranking.get("discovery_score") or 0.0) + 0.12 * float(transcript_ranking.get("concept_match") or 0.0),
                    )
                    ranking["clipability_score"] = min(
                        1.0,
                        float(ranking.get("clipability_score") or 0.0)
                        + 0.18 * float(transcript_ranking.get("clipability_signal") or 0.0),
                    )

                segments: list[SegmentMatch] = []
                if transcript:
                    if fast_mode:
                        segments = self._fast_segments_from_transcript(
                            transcript=transcript,
                            concept_terms=concept_terms,
                            max_segments=2,
                        )
                    else:
                        chunks, chunk_embeddings = self._load_or_create_transcript_chunks(conn, video_id, transcript)
                        if chunks and len(chunk_embeddings) > 0:
                            if concept_embedding is not None:
                                segments = select_segments(
                                    concept_embedding,
                                    chunk_embeddings,
                                    chunks,
                                    concept_terms=concept_terms,
                                    top_k=6 if vague_topic else 4,
                                )
                            else:
                                segments = self._fast_segments_from_transcript(
                                    transcript=transcript,
                                    concept_terms=concept_terms,
                                    max_segments=3,
                                )
                    if not segments:
                        segments = self._fallback_segments_from_transcript(transcript)
                else:
                    metadata_segment = self._fallback_segment_from_video_metadata(
                        video,
                        concept_terms,
                        target_clip_duration_sec=safe_target_clip_duration,
                    )
                    if metadata_segment:
                        segments = [metadata_segment]
                    elif not use_full_short_clip:
                        continue

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
                if not segment_candidates and str(video.get("search_source") or "") == "local_cache" and segments:
                    # Local-corpus recovery can miss lexical gates for typo/noisy concepts.
                    seed = dict(candidate.get("video_relevance") or {})
                    seed["passes"] = True
                    seed["score"] = max(0.08, float(seed.get("score") or 0.0))
                    segment_candidates = [(segments[0], seed)]
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
                            min_len=clip_min_len,
                            max_len=clip_max_len,
                        )
                    else:
                        clip_window = self._normalize_clip_window(
                            segment.t_start,
                            segment.t_end,
                            video_duration,
                            min_len=clip_min_len,
                            max_len=clip_max_len,
                        )
                    if not clip_window:
                        continue
                    start_sec, end_sec = clip_window
                    clip_key = self._clip_key(video_id, start_sec, end_sec)
                    if clip_key in existing_clip_keys or clip_key in generated_clip_keys:
                        continue

                    relevance_context = self._merge_relevance_context(
                        candidate.get("video_relevance") or {},
                        segment_relevance,
                    )
                    relevance_context["query_strategy"] = query_candidate.strategy
                    relevance_context["retrieval_stage"] = str(candidate.get("stage") or "")
                    relevance_context["discovery_score"] = float(ranking.get("discovery_score") or 0.0)
                    relevance_context["clipability_score"] = float(ranking.get("clipability_score") or 0.0)
                    relevance_context["source_surface"] = str(video.get("search_source") or "")
                    relevance_context["score"] = (
                        0.58 * float(relevance_context.get("score") or 0.0)
                        + 0.28 * float(ranking.get("discovery_score") or 0.0)
                        + 0.14 * float(ranking.get("clipability_score") or 0.0)
                    )
                    if not bool(relevance_context.get("passes", True)):
                        continue

                    if dry_run:
                        preview = self._build_dry_run_reel_preview(
                            concept=concept,
                            video=video,
                            segment=segment,
                            clip_window=clip_window,
                            relevance_context=relevance_context,
                        )
                        generated.append(preview)
                    else:
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
                            target_clip_duration_sec=safe_target_clip_duration,
                        )
                        if not reel:
                            continue
                        generated.append(reel)

                    generated_clip_keys.add(clip_key)
                    generated_video_counts[video_id] = generated_video_counts.get(video_id, 0) + 1
                    selected_outcome = {
                        "video_id": video_id,
                        "reasons": [
                            f"query_strategy:{query_candidate.strategy}",
                            f"stage:{candidate.get('stage')}",
                            f"discovery:{round(float(ranking.get('discovery_score') or 0.0), 3)}",
                            f"clipability:{round(float(ranking.get('clipability_score') or 0.0), 3)}",
                        ],
                        "clip_window": {"t_start": start_sec, "t_end": end_sec},
                    }

                    if self._should_finalize_generation(
                        generated=generated,
                        num_reels=num_reels,
                        preferred_video_duration=safe_video_duration_pref,
                        max_generation_target=max_generation_target,
                    ):
                        self._persist_retrieval_debug_run(
                            conn,
                            run=retrieval_run,
                            query_reports=all_query_reports,
                            candidate_rows=candidate_records,
                            selected=selected_outcome,
                            failure_reason="",
                            dry_run=dry_run,
                        )
                        return self._finalize_generated_reels(
                            generated=generated,
                            num_reels=num_reels,
                            preferred_video_duration=safe_video_duration_pref,
                        )
                    if existing_for_video + generated_video_counts.get(video_id, 0) >= max_segments_per_video:
                        break

            self._persist_retrieval_debug_run(
                conn,
                run=retrieval_run,
                query_reports=all_query_reports,
                candidate_rows=candidate_records,
                selected=selected_outcome,
                failure_reason="" if selected_outcome else "no_segment_after_rerank",
                dry_run=dry_run,
            )
        return self._finalize_generated_reels(
            generated=generated,
            num_reels=num_reels,
            preferred_video_duration=safe_video_duration_pref,
        )

    def _normalize_video_pool_mode(self, value: str | None) -> str:
        if value in self.VALID_VIDEO_POOL_MODES:
            return str(value)
        return "short-first"

    def _normalize_preferred_video_duration(self, value: str | None) -> str:
        if value in self.VALID_VIDEO_DURATION_PREFS:
            return str(value)
        return "any"

    def _normalize_target_clip_duration(self, value: int | float | None) -> int:
        if value is None:
            return self.DEFAULT_TARGET_CLIP_DURATION_SEC
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return self.DEFAULT_TARGET_CLIP_DURATION_SEC
        return max(self.MIN_TARGET_CLIP_DURATION_SEC, min(self.MAX_TARGET_CLIP_DURATION_SEC, parsed))

    def _duration_plan(self, video_pool_mode: str, preferred_video_duration: str) -> tuple[str | None, ...]:
        if preferred_video_duration == "short":
            return ("short", "medium", "long", None)
        if preferred_video_duration == "medium":
            return ("medium", "long", "short", None)
        if preferred_video_duration == "long":
            return ("long", "medium", "short", None)
        if video_pool_mode == "long-form":
            return ("long", "medium", "short", None)
        if video_pool_mode == "balanced":
            return ("short", "long", "medium", None)
        return ("short", "long", "medium", None)

    def _search_passes(self, video_pool_mode: str, preferred_video_duration: str) -> list[dict[str, Any]]:
        if preferred_video_duration in {"short", "medium", "long"}:
            return [
                {
                    "duration_plan": (preferred_video_duration,),
                    "strict_duration": True,
                    "allow_unknown_duration": False,
                },
                {
                    "duration_plan": self._duration_plan(video_pool_mode, preferred_video_duration),
                    "strict_duration": False,
                    "allow_unknown_duration": True,
                },
            ]
        return [
            {
                "duration_plan": self._duration_plan(video_pool_mode, preferred_video_duration),
                "strict_duration": False,
                "allow_unknown_duration": True,
            }
        ]

    def _video_matches_preferred_duration(
        self,
        video_duration_sec: int,
        preferred_video_duration: str,
        allow_unknown_duration: bool,
    ) -> bool:
        if preferred_video_duration not in {"short", "medium", "long"}:
            return True
        if video_duration_sec <= 0:
            return allow_unknown_duration
        if preferred_video_duration == "short":
            return video_duration_sec < 4 * 60
        if preferred_video_duration == "medium":
            return 4 * 60 <= video_duration_sec <= 20 * 60
        return video_duration_sec > 20 * 60

    def _video_duration_bucket(self, duration_sec: int | float | None) -> str | None:
        try:
            parsed = int(duration_sec or 0)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        if parsed < 4 * 60:
            return "short"
        if parsed <= 20 * 60:
            return "medium"
        return "long"

    def _generation_result_score(self, reel: dict[str, Any]) -> float:
        relevance = reel.get("relevance_score")
        if isinstance(relevance, (int, float)):
            return float(relevance)
        score = reel.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    def _has_short_and_long_mix(self, generated: list[dict[str, Any]]) -> bool:
        has_short = False
        has_long = False
        for reel in generated:
            bucket = self._video_duration_bucket(reel.get("video_duration_sec"))
            if bucket == "short":
                has_short = True
            elif bucket == "long":
                has_long = True
            if has_short and has_long:
                return True
        return False

    def _generation_target_cap(
        self,
        num_reels: int,
        preferred_video_duration: str,
        fast_mode: bool,
    ) -> int:
        if preferred_video_duration != "any" or num_reels <= 1:
            return num_reels
        extra = max(2, min(8, num_reels // 2 + (1 if fast_mode else 2)))
        return num_reels + extra

    def _search_results_budget(
        self,
        *,
        fast_mode: bool,
        generated_count: int,
        max_generation_target: int,
    ) -> int:
        remaining = max(1, max_generation_target - max(0, generated_count))
        if self.serverless_mode:
            return max(6, min(10, 4 + remaining))
        if fast_mode:
            return max(6, min(12, 5 + remaining))
        return max(18, min(42, 14 + remaining * 4))

    def _transcript_expansion_budget(
        self,
        *,
        fast_mode: bool,
        generated_count: int,
        max_generation_target: int,
    ) -> int:
        remaining = max(1, max_generation_target - max(0, generated_count))
        if self.serverless_mode:
            return max(1, min(3, remaining))
        if fast_mode:
            return max(2, min(4, remaining))
        return max(4, min(12, remaining * 2))

    def _should_finalize_generation(
        self,
        generated: list[dict[str, Any]],
        num_reels: int,
        preferred_video_duration: str,
        max_generation_target: int,
    ) -> bool:
        if len(generated) < num_reels:
            return False
        if preferred_video_duration != "any" or num_reels <= 1:
            return True
        if self._has_short_and_long_mix(generated):
            return True
        return len(generated) >= max_generation_target

    def _finalize_generated_reels(
        self,
        generated: list[dict[str, Any]],
        num_reels: int,
        preferred_video_duration: str,
    ) -> list[dict[str, Any]]:
        if not generated or num_reels <= 0:
            return []
        if preferred_video_duration != "any" or num_reels <= 1:
            return generated[:num_reels]

        ranked = sorted(generated, key=self._generation_result_score, reverse=True)
        short_candidates = [reel for reel in ranked if self._video_duration_bucket(reel.get("video_duration_sec")) == "short"]
        long_candidates = [reel for reel in ranked if self._video_duration_bucket(reel.get("video_duration_sec")) == "long"]

        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for candidate in [short_candidates[0] if short_candidates else None, long_candidates[0] if long_candidates else None]:
            if not candidate:
                continue
            reel_id = str(candidate.get("reel_id") or "")
            if reel_id and reel_id in seen_ids:
                continue
            selected.append(candidate)
            if reel_id:
                seen_ids.add(reel_id)

        for candidate in ranked:
            if len(selected) >= num_reels:
                break
            reel_id = str(candidate.get("reel_id") or "")
            if reel_id and reel_id in seen_ids:
                continue
            selected.append(candidate)
            if reel_id:
                seen_ids.add(reel_id)

        selected = selected[:num_reels]
        selected.sort(key=self._generation_result_score, reverse=True)
        return selected

    def _parse_keywords_json(self, value: Any) -> list[str]:
        raw = str(value or "[]")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        if not isinstance(parsed, list):
            return []
        clean: list[str] = []
        for item in parsed:
            token = str(item or "").strip()
            if not token:
                continue
            clean.append(token)
            if len(clean) >= 12:
                break
        return clean

    def _clip_length_bounds(self, target_clip_duration_sec: int) -> tuple[int, int]:
        safe_target = self._normalize_target_clip_duration(target_clip_duration_sec)
        min_len = max(10, int(round(safe_target * 0.35)))
        max_len = max(min_len + self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC, safe_target)
        return min_len, max_len

    def _resolve_clip_duration_bounds(
        self,
        target_clip_duration_sec: int,
        target_clip_duration_min_sec: int | None,
        target_clip_duration_max_sec: int | None,
    ) -> tuple[int, int, int]:
        safe_target = self._normalize_target_clip_duration(target_clip_duration_sec)
        default_min, default_max = self._clip_length_bounds(safe_target)

        if target_clip_duration_min_sec is None and target_clip_duration_max_sec is None:
            return default_min, default_max, safe_target

        safe_min = default_min if target_clip_duration_min_sec is None else self._normalize_target_clip_duration(
            target_clip_duration_min_sec
        )
        safe_max = default_max if target_clip_duration_max_sec is None else self._normalize_target_clip_duration(
            target_clip_duration_max_sec
        )
        if safe_min > safe_max:
            safe_min, safe_max = safe_max, safe_min
        if safe_max - safe_min < self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC:
            expanded_max = min(self.MAX_TARGET_CLIP_DURATION_SEC, safe_min + self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC)
            if expanded_max - safe_min >= self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC:
                safe_max = expanded_max
            else:
                safe_min = max(self.MIN_TARGET_CLIP_DURATION_SEC, safe_max - self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC)
        safe_target = max(safe_min, min(safe_max, safe_target))
        return safe_min, safe_max, safe_target

    def _get_concept_embedding(self, conn, concept: dict[str, Any]) -> np.ndarray:
        if concept.get("embedding_json"):
            return np.array(json.loads(concept["embedding_json"]), dtype=np.float32)

        parsed_keywords = self._parse_keywords_json(concept.get("keywords_json"))
        concept_text = (
            f"{concept['title']}. "
            f"Keywords: {' '.join(parsed_keywords)}. "
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

    def _init_retrieval_debug_run(self, material_id: str, concept_id: str, concept_title: str) -> dict[str, Any]:
        return {
            "run_id": str(uuid.uuid4()),
            "material_id": material_id,
            "concept_id": concept_id,
            "concept_title": concept_title,
        }

    def _persist_retrieval_debug_run(
        self,
        conn,
        run: dict[str, Any],
        query_reports: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        selected: dict[str, Any] | None,
        failure_reason: str,
        dry_run: bool,
    ) -> None:
        if not self.retrieval_debug_logging:
            return
        run_id = str(run.get("run_id") or "").strip()
        if not run_id:
            return

        upsert(
            conn,
            "retrieval_runs",
            {
                "id": run_id,
                "material_id": str(run.get("material_id") or ""),
                "concept_id": str(run.get("concept_id") or ""),
                "concept_title": str(run.get("concept_title") or ""),
                "selected_video_id": str((selected or {}).get("video_id") or ""),
                "failure_reason": failure_reason.strip(),
                "debug_json": dumps_json(
                    {
                        "selected": selected,
                        "query_count": len(query_reports),
                        "candidate_count": len(candidate_rows),
                        "dry_run": bool(dry_run),
                    }
                ),
                "created_at": now_iso(),
            },
        )

        for idx, row in enumerate(query_reports[:240]):
            upsert(
                conn,
                "retrieval_queries",
                {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "query_text": str(row.get("query") or ""),
                    "strategy": str(row.get("strategy") or ""),
                    "stage": str(row.get("stage") or ""),
                    "source_surface": str(row.get("surface") or ""),
                    "source_terms_json": dumps_json(row.get("source_terms") or []),
                    "weight": float(row.get("weight") or 0.0),
                    "result_count": int(row.get("results") or 0),
                    "kept_count": int(row.get("kept") or 0),
                    "position": idx,
                    "created_at": now_iso(),
                },
            )

        for idx, row in enumerate(candidate_rows[:320]):
            upsert(
                conn,
                "retrieval_candidates",
                {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "video_id": str(row.get("video_id") or ""),
                    "video_title": str(row.get("video_title") or ""),
                    "channel_title": str(row.get("channel_title") or ""),
                    "strategy": str(row.get("strategy") or ""),
                    "stage": str(row.get("stage") or ""),
                    "query_text": str(row.get("query") or ""),
                    "source_surface": str(row.get("source_surface") or ""),
                    "discovery_score": float(row.get("discovery_score") or 0.0),
                    "clipability_score": float(row.get("clipability_score") or 0.0),
                    "final_score": float(row.get("final_score") or 0.0),
                    "feature_json": dumps_json(row.get("features") or {}),
                    "position": idx,
                    "created_at": now_iso(),
                },
            )

        if selected:
            selected_video_id = str(selected.get("video_id") or "")
            can_write_outcome = not dry_run
            if not can_write_outcome and selected_video_id:
                can_write_outcome = bool(fetch_one(conn, "SELECT id FROM videos WHERE id = ?", (selected_video_id,)))
            if not can_write_outcome:
                return
            clip_window = selected.get("clip_window") or {}
            upsert(
                conn,
                "retrieval_outcomes",
                {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "video_id": selected_video_id,
                    "t_start": float(clip_window.get("t_start") or 0.0),
                    "t_end": float(clip_window.get("t_end") or 0.0),
                    "reason_json": dumps_json(selected.get("reasons") or []),
                    "created_at": now_iso(),
                },
            )

    def _derive_visual_scene_spec(self, title: str, keywords: list[str], summary: str) -> dict[str, list[str]]:
        raw = " ".join([title, summary, *keywords]).strip().lower()
        noun_like = [t for t in normalize_terms([raw]) if len(t) >= 4][:18]
        actions: list[str] = []
        for token in noun_like:
            if token.endswith("ing") or token in {"show", "build", "explain", "demo", "review"}:
                actions.append(token)
        if not actions:
            actions = ["showing", "demonstration"]

        return {
            "environment": noun_like[:5],
            "subjects": noun_like[5:10],
            "objects": noun_like[10:16],
            "actions": actions[:5],
            "camera": ["close up", "b-roll", "cinematic"],
            "mood": ["clean", "focused"],
        }

    def _build_query_candidates(
        self,
        title: str,
        keywords: list[str],
        summary: str,
        subject_tag: str | None,
        context_terms: list[str],
        visual_spec: dict[str, list[str]],
        fast_mode: bool,
    ) -> list[QueryCandidate]:
        clean_title = " ".join(str(title or "").split()).strip()
        if not clean_title:
            return []

        subject = " ".join(str(subject_tag or "").split()).strip()
        keyword_terms = [k.strip() for k in keywords if k.strip()][:8]
        context_terms = [t.strip() for t in context_terms if t.strip()][:8]
        primary_terms = [clean_title, *keyword_terms[:4], *context_terms[:3]]

        scene_terms = [
            *visual_spec.get("environment", []),
            *visual_spec.get("subjects", []),
            *visual_spec.get("objects", []),
        ]
        action_terms = visual_spec.get("actions", [])

        candidates: list[QueryCandidate] = []

        def add_candidate(
            text: str,
            strategy: str,
            confidence: float,
            source_terms: list[str],
            stage: str,
            weight: float = 1.0,
            source_surface: str = "youtube_api",
        ) -> None:
            cleaned = " ".join(text.split()).strip()
            if not cleaned:
                return
            candidates.append(
                QueryCandidate(
                    text=cleaned,
                    strategy=strategy,
                    confidence=max(0.0, min(1.0, confidence)),
                    source_terms=source_terms[:8],
                    weight=max(0.05, float(weight)),
                    stage=stage,
                    source_surface=source_surface,
                )
            )

        core = " ".join(part for part in [subject, clean_title] if part).strip()
        add_candidate(core, "literal", 0.96, [clean_title, subject], "high_precision", 1.0)
        if keyword_terms:
            add_candidate(f"{core} {' '.join(keyword_terms[:2])}", "literal", 0.9, keyword_terms[:2], "high_precision", 0.95)

        paraphrases = self._expand_controlled_synonyms([clean_title, *keyword_terms[:3]])
        for phrase in paraphrases[: (3 if fast_mode else 8)]:
            add_candidate(f"{subject} {phrase}".strip(), "paraphrase", 0.74, [phrase], "broad", 0.84)

        if scene_terms:
            add_candidate(
                f"{' '.join(scene_terms[:3])} footage",
                "scene",
                0.86,
                scene_terms[:3],
                "high_precision",
                0.95,
            )
            add_candidate(
                f"{' '.join(scene_terms[:3])} b-roll cinematic 4k",
                "broll",
                0.88,
                scene_terms[:3],
                "high_precision",
                0.98,
            )

        if action_terms:
            add_candidate(
                f"{' '.join(action_terms[:2])} {' '.join(scene_terms[:2])} footage",
                "action",
                0.77,
                [*action_terms[:2], *scene_terms[:2]],
                "broad",
                0.82,
            )

        if scene_terms:
            add_candidate(
                f"{' '.join(scene_terms[:3])} report interview coverage archive",
                "news_doc",
                0.69,
                scene_terms[:3],
                "broad",
                0.74,
            )
            add_candidate(
                f"{clean_title} how it works demonstration example",
                "tutorial_demo",
                0.72,
                [clean_title],
                "broad",
                0.76,
            )

        if self.retrieval_tier2_enabled:
            for recovery in self._decompose_concept_for_recovery(clean_title, summary, keyword_terms)[: (3 if fast_mode else 7)]:
                add_candidate(
                    f"{recovery} footage scene",
                    "recovery_adjacent",
                    0.62,
                    [recovery],
                    "recovery",
                    0.66,
                    source_surface="duckduckgo_site",
                )
                add_candidate(
                    f"\"{recovery}\" site:youtube.com/watch",
                    "recovery_adjacent",
                    0.6,
                    [recovery],
                    "recovery",
                    0.62,
                    source_surface="duckduckgo_quoted",
                )

        deduped: list[QueryCandidate] = []
        seen: set[str] = set()
        ordered = sorted(
            candidates,
            key=lambda item: (item.stage != "high_precision", -(item.confidence * item.weight), len(item.text)),
        )
        for item in ordered:
            key = item.text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= (18 if fast_mode else 42):
                break
        return deduped

    def _build_retrieval_stage_plan(
        self,
        query_candidates: list[QueryCandidate],
        fast_mode: bool,
    ) -> list[RetrievalStagePlan]:
        stage_map: dict[str, list[QueryCandidate]] = {"high_precision": [], "broad": [], "recovery": []}
        for item in query_candidates:
            stage = item.stage if item.stage in stage_map else "broad"
            stage_map[stage].append(item)

        if self.serverless_mode:
            high_precision_budget = 2
            high_precision_min = 2
            broad_budget = 2
            broad_min = 2
            recovery_budget = 1
            recovery_min = 1
        else:
            high_precision_budget = 3 if fast_mode else 5
            high_precision_min = 3 if fast_mode else 8
            broad_budget = 5 if fast_mode else 14
            broad_min = 4 if fast_mode else 12
            recovery_budget = 3 if fast_mode else 8
            recovery_min = 2 if fast_mode else 6

        plans = [
            RetrievalStagePlan(
                name="high_precision",
                queries=sorted(stage_map["high_precision"], key=lambda q: -(q.confidence * q.weight)),
                budget=high_precision_budget,
                min_good_results=high_precision_min,
            ),
            RetrievalStagePlan(
                name="broad",
                queries=sorted(stage_map["broad"], key=lambda q: -(q.confidence * q.weight)),
                budget=broad_budget,
                min_good_results=broad_min,
            ),
            RetrievalStagePlan(
                name="recovery",
                queries=sorted(stage_map["recovery"], key=lambda q: -(q.confidence * q.weight)),
                budget=recovery_budget,
                min_good_results=recovery_min,
            ),
        ]
        return [plan for plan in plans if plan.queries and plan.budget > 0]

    def _fast_pass_is_sufficient(
        self,
        stage_candidates: list[dict[str, Any]],
        *,
        fast_mode: bool,
        max_generation_target: int,
    ) -> bool:
        if not stage_candidates:
            return False
        if fast_mode:
            target = max(6, min(14, max_generation_target + 2))
        else:
            target = max(8, min(40, max_generation_target * 3))
        if len(stage_candidates) < target:
            return False
        strong = sum(
            1
            for candidate in stage_candidates
            if float((candidate.get("ranking") or {}).get("discovery_score") or 0.0) >= 0.2
        )
        return strong >= (max(2, target // 4) if fast_mode else max(4, target // 3))

    def _stage_search_jobs_parallel(
        self,
        *,
        stage_name: str,
        stage_queries: list[QueryCandidate],
        stage_duration_plan: tuple[str | None, ...],
        max_results_for_query: int,
        creative_commons_only: bool,
        fast_mode: bool,
    ) -> list[tuple[int, int, QueryCandidate, str | None, list[dict[str, Any]]]]:
        jobs: list[tuple[int, int, QueryCandidate, str | None]] = []
        for query_idx, query_candidate in enumerate(stage_queries):
            for duration_idx, duration in enumerate(stage_duration_plan):
                jobs.append((query_idx, duration_idx, query_candidate, duration))
        if not jobs:
            return []

        workers_limit = self.QUERY_RETRIEVAL_WORKERS_FAST if fast_mode else self.QUERY_RETRIEVAL_WORKERS_SLOW
        if self.serverless_mode:
            workers_limit = min(workers_limit, 2)
        workers = max(1, min(workers_limit, len(jobs)))
        if workers == 1:
            output: list[tuple[int, int, QueryCandidate, str | None, list[dict[str, Any]]]] = []
            for query_idx, duration_idx, query_candidate, duration in jobs:
                videos = self.youtube_service.search_videos(
                    None,
                    query=query_candidate.text,
                    max_results=max_results_for_query,
                    creative_commons_only=creative_commons_only,
                    video_duration=duration,
                    retrieval_strategy=query_candidate.strategy,
                    retrieval_stage=stage_name,
                    source_surface=query_candidate.source_surface,
                )
                output.append((query_idx, duration_idx, query_candidate, duration, videos))
            return output

        output: list[tuple[int, int, QueryCandidate, str | None, list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    self.youtube_service.search_videos,
                    None,
                    query_candidate.text,
                    max_results_for_query,
                    creative_commons_only,
                    duration,
                    query_candidate.strategy,
                    stage_name,
                    query_candidate.source_surface,
                ): (query_idx, duration_idx, query_candidate, duration)
                for query_idx, duration_idx, query_candidate, duration in jobs
            }
            for future in as_completed(future_map):
                query_idx, duration_idx, query_candidate, duration = future_map[future]
                try:
                    videos = future.result()
                except Exception:
                    videos = []
                output.append((query_idx, duration_idx, query_candidate, duration, videos))

        output.sort(key=lambda row: (row[0], row[1]))
        return output

    def _prefetch_transcripts_parallel(
        self,
        video_ids: list[str],
        *,
        fast_mode: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        unique_ids = [str(video_id).strip() for video_id in dict.fromkeys(video_ids) if str(video_id).strip()]
        if not unique_ids:
            return {}

        workers_limit = self.TRANSCRIPT_FETCH_WORKERS_FAST if fast_mode else self.TRANSCRIPT_FETCH_WORKERS_SLOW
        if self.serverless_mode:
            workers_limit = min(workers_limit, 2)
        workers = max(1, min(workers_limit, len(unique_ids)))
        if workers == 1:
            return {video_id: self.youtube_service.get_transcript(None, video_id) for video_id in unique_ids}

        transcripts: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self.youtube_service.get_transcript, None, video_id): video_id for video_id in unique_ids
            }
            for future in as_completed(future_map):
                video_id = future_map[future]
                try:
                    transcript = future.result()
                except Exception:
                    transcript = []
                transcripts[video_id] = transcript
        return transcripts

    def _stage_duration_plan(
        self,
        stage_name: str,
        preferred_video_duration: str,
        video_pool_mode: str,
        fast_mode: bool,
    ) -> tuple[str | None, ...]:
        if stage_name == "high_precision":
            if preferred_video_duration in {"short", "medium", "long"}:
                return (preferred_video_duration,)
            return ("short", None)
        if stage_name == "broad":
            if fast_mode:
                if preferred_video_duration in {"short", "medium", "long"}:
                    return (preferred_video_duration, None)
                if video_pool_mode == "long-form":
                    return ("long", None)
                if video_pool_mode == "balanced":
                    return ("short", "long")
                return ("short", None)
            return self._duration_plan(video_pool_mode, preferred_video_duration)
        if fast_mode:
            if preferred_video_duration in {"short", "medium", "long"}:
                return (preferred_video_duration, None)
            return (None, "short")
        return (None, "short", "medium", "long")

    def _score_video_candidate(
        self,
        conn,
        video: dict[str, Any],
        query_candidate: QueryCandidate,
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        visual_spec: dict[str, list[str]],
        preferred_video_duration: str,
        stage_name: str,
        require_context: bool,
        fast_mode: bool,
    ) -> dict[str, Any]:
        title = str(video.get("title") or "")
        description = str(video.get("description") or "")
        metadata_text = self._video_metadata_text(video)
        text_relevance = self._score_text_relevance(
            conn,
            text=metadata_text,
            concept_terms=concept_terms,
            context_terms=context_terms,
            concept_embedding=concept_embedding,
            subject_tag=subject_tag,
        )

        semantic_title = self._semantic_similarity(
            conn,
            text=title,
            concept_terms=concept_terms,
            concept_embedding=concept_embedding,
        )
        semantic_description = self._semantic_similarity(
            conn,
            text=description,
            concept_terms=concept_terms,
            concept_embedding=concept_embedding,
        )
        strategy_prior = float(self.QUERY_STRATEGY_PRIOR.get(query_candidate.strategy, 0.66))
        strategy_prior *= self._learned_strategy_factor(conn, query_candidate.strategy)
        stage_prior = 0.1 if stage_name == "high_precision" else (0.05 if stage_name == "broad" else 0.0)
        duration_fit = self._duration_fit_score(
            duration_sec=int(video.get("duration_sec") or 0),
            preferred_video_duration=preferred_video_duration,
        )
        freshness_fit = self._freshness_fit_score(video.get("published_at"))
        channel_quality = self._channel_quality_score(video)
        visual_intent_match = lexical_overlap_score(
            f"{title} {description}",
            [
                *visual_spec.get("environment", []),
                *visual_spec.get("objects", []),
                *visual_spec.get("actions", []),
            ],
        )
        source_prior = float(self.SOURCE_SURFACE_PRIOR.get(str(video.get("search_source") or ""), 0.82))
        clipability = self._score_clipability_from_metadata(video, strategy=query_candidate.strategy)
        engagement_fit = self._engagement_fit_score(video.get("view_count"))
        discovery = (
            0.28 * semantic_title
            + 0.18 * semantic_description
            + 0.10 * strategy_prior
            + 0.09 * duration_fit
            + 0.08 * freshness_fit
            + 0.08 * channel_quality
            + 0.09 * engagement_fit
            + 0.10 * visual_intent_match
            + stage_prior
        )
        discovery *= source_prior
        discovery = float(max(0.0, min(1.0, discovery)))

        text_pass = self._passes_relevance_gate(
            relevance=text_relevance,
            require_context=require_context,
            fast_mode=fast_mode,
        )
        min_discovery = 0.14 if fast_mode else 0.16
        passes = text_pass and discovery >= min_discovery

        final_score = 0.74 * discovery + 0.26 * clipability
        text_relevance = dict(text_relevance)
        text_relevance["passes"] = passes
        return {
            "passes": passes,
            "final_score": float(final_score),
            "discovery_score": discovery,
            "clipability_score": float(clipability),
            "text_relevance": text_relevance,
            "features": {
                "semantic_title": float(semantic_title),
                "semantic_description": float(semantic_description),
                "strategy_prior": strategy_prior,
                "duration_fit": float(duration_fit),
                "freshness_fit": float(freshness_fit),
                "channel_quality": float(channel_quality),
                "engagement_fit": float(engagement_fit),
                "visual_intent_match": float(visual_intent_match),
                "source_prior": float(source_prior),
            },
        }

    def _engagement_fit_score(self, view_count: Any) -> float:
        try:
            views = int(view_count or 0)
        except (TypeError, ValueError):
            views = 0
        if views <= 0:
            return 0.55
        if views >= 5_000_000:
            return 0.95
        if views >= 1_000_000:
            return 0.9
        if views >= 300_000:
            return 0.84
        if views >= 100_000:
            return 0.78
        if views >= 30_000:
            return 0.7
        if views >= 10_000:
            return 0.64
        return 0.58

    def _semantic_similarity(
        self,
        conn,
        text: str,
        concept_terms: list[str],
        concept_embedding: np.ndarray | None,
    ) -> float:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return 0.0
        lexical = lexical_overlap_score(cleaned, concept_terms)
        if concept_embedding is None:
            return float(lexical)
        try:
            text_embedding = self.embedding_service.embed_texts(conn, [cleaned])[0]
            semantic = float(text_embedding @ concept_embedding.astype(np.float32))
        except Exception:
            semantic = 0.0
        return float(max(0.0, min(1.0, 0.82 * max(0.0, semantic) + 0.18 * lexical)))

    def _duration_fit_score(self, duration_sec: int, preferred_video_duration: str) -> float:
        if duration_sec <= 0:
            return 0.55
        bucket = self._video_duration_bucket(duration_sec)
        if preferred_video_duration in {"short", "medium", "long"}:
            return 0.95 if bucket == preferred_video_duration else 0.35
        if bucket == "short":
            return 0.88
        if bucket == "medium":
            return 0.72
        return 0.5

    def _freshness_fit_score(self, published_at: Any) -> float:
        published = str(published_at or "").strip()
        if not published:
            return 0.58
        try:
            year = int(published[:4])
        except (TypeError, ValueError):
            return 0.58
        if year >= 2024:
            return 0.84
        if year >= 2020:
            return 0.72
        if year >= 2015:
            return 0.6
        return 0.46

    def _channel_quality_score(self, video: dict[str, Any]) -> float:
        title = str(video.get("title") or "").lower()
        channel = str(video.get("channel_title") or "").lower()
        bucket = self._infer_channel_tier(channel=channel, title=title)
        bonus = float(self.CHANNEL_QUALITY_BONUS.get(bucket, 0.0))
        return float(max(0.0, min(1.0, 0.6 + bonus)))

    def _learned_strategy_factor(self, conn, strategy: str) -> float:
        key = str(strategy or "").strip().lower() or "literal"
        cached = self._strategy_history_cache.get(key)
        if cached is not None:
            return cached

        try:
            row = fetch_one(
                conn,
                """
                SELECT
                    AVG(CASE WHEN result_count > 0 THEN CAST(kept_count AS REAL) / result_count ELSE 0 END) AS kept_ratio
                FROM (
                    SELECT result_count, kept_count
                    FROM retrieval_queries
                    WHERE strategy = ?
                    ORDER BY created_at DESC
                    LIMIT 240
                ) q
                """,
                (key,),
            )
        except Exception:
            row = None

        kept_ratio = float((row or {}).get("kept_ratio") or 0.0)
        # Map historical keep ratio into a soft multiplicative factor.
        factor = max(0.85, min(1.18, 0.9 + 0.45 * kept_ratio))
        self._strategy_history_cache[key] = factor
        return factor

    def _infer_channel_tier(self, channel: str, title: str) -> str:
        hay = f"{channel} {title}"
        if any(token in hay for token in ["news", "times", "bbc", "cnn", "reuters", "al jazeera", "pbs"]):
            return "news"
        if any(token in hay for token in ["academy", "course", "university", "education", "science", "explained"]):
            return "education"
        if any(token in hay for token in ["stock", "footage", "cinematic", "film", "camera"]):
            return "stock_footage"
        if any(token in hay for token in ["tutorial", "how to", "walkthrough", "demo"]):
            return "tutorial"
        if any(token in hay for token in ["compilation", "best moments", "reaction", "clips"]):
            return "low_quality_compilation"
        if "podcast" in hay:
            return "podcast"
        return "tutorial"

    def _score_clipability_from_metadata(self, video: dict[str, Any], strategy: str) -> float:
        duration_sec = int(video.get("duration_sec") or 0)
        title = str(video.get("title") or "").lower()
        description = str(video.get("description") or "").lower()

        if duration_sec <= 0:
            duration_score = 0.58
        elif duration_sec <= 75:
            duration_score = 0.9
        elif duration_sec <= 6 * 60:
            duration_score = 0.82
        elif duration_sec <= 18 * 60:
            duration_score = 0.7
        else:
            duration_score = 0.5

        penalty = 0.0
        metadata_text = f"{title} {description}"
        for token in self.CLIPABILITY_PENALTY_TOKENS:
            if token in metadata_text:
                penalty += 0.07
        strategy_boost = 0.08 if strategy in {"scene", "action", "broll"} else 0.0
        return float(max(0.0, min(1.0, duration_score + strategy_boost - penalty)))

    def _score_transcript_alignment(
        self,
        conn,
        transcript: list[dict[str, Any]],
        concept_terms: list[str],
        concept_embedding: np.ndarray | None,
        visual_spec: dict[str, list[str]],
    ) -> dict[str, float]:
        if not transcript:
            return {"concept_match": 0.0, "clipability_signal": 0.0}

        excerpt_parts: list[str] = []
        first_window = []
        for entry in transcript[:120]:
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            excerpt_parts.append(text)
            if float(entry.get("start") or 0.0) <= 70:
                first_window.append(text)
            if len(excerpt_parts) >= 80:
                break
        excerpt = " ".join(excerpt_parts).strip()
        early_excerpt = " ".join(first_window).strip()

        concept_match = self._semantic_similarity(
            conn,
            text=excerpt,
            concept_terms=concept_terms,
            concept_embedding=concept_embedding,
        )
        early_signal = lexical_overlap_score(early_excerpt, concept_terms) if early_excerpt else 0.0
        visual_match = lexical_overlap_score(
            excerpt,
            [
                *visual_spec.get("environment", []),
                *visual_spec.get("objects", []),
                *visual_spec.get("actions", []),
            ],
        )
        dense_segments = 0
        for entry in transcript[:120]:
            text = str(entry.get("text") or "").strip()
            if len(text.split()) >= 4:
                dense_segments += 1
        density = min(1.0, dense_segments / max(1, min(60, len(transcript))))
        clipability_signal = 0.5 * density + 0.25 * early_signal + 0.25 * visual_match
        return {
            "concept_match": float(max(0.0, min(1.0, concept_match))),
            "clipability_signal": float(max(0.0, min(1.0, clipability_signal))),
        }

    def _collapse_near_duplicate_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        ordered = sorted(
            candidates,
            key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        for row in ordered:
            duplicate = False
            for prev in kept:
                if self._video_candidates_near_duplicate(row, prev):
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(row)
        return kept

    def _video_candidates_near_duplicate(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        lv = left.get("video") or {}
        rv = right.get("video") or {}
        lt = self._normalize_title_for_similarity(str(lv.get("title") or ""))
        rt = self._normalize_title_for_similarity(str(rv.get("title") or ""))
        if not lt or not rt:
            return False
        title_sim = self._token_jaccard(lt, rt)
        ld = int(lv.get("duration_sec") or 0)
        rd = int(rv.get("duration_sec") or 0)
        close_duration = ld > 0 and rd > 0 and abs(ld - rd) <= 5
        same_channel = str(lv.get("channel_title") or "").strip().lower() == str(rv.get("channel_title") or "").strip().lower()
        return title_sim >= 0.86 or (title_sim >= 0.72 and close_duration and same_channel)

    def _normalize_title_for_similarity(self, text: str) -> set[str]:
        tokens = normalize_terms([text])
        return {t for t in tokens if t not in {"video", "official", "footage", "clip", "shorts"}}

    def _token_jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left.intersection(right))
        union = len(left.union(right))
        return float(overlap / max(1, union))

    def _diversify_video_candidates(self, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not candidates:
            return []
        remaining = sorted(
            candidates,
            key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = -1.0
            for idx, cand in enumerate(remaining):
                relevance = float((cand.get("ranking") or {}).get("final_score") or 0.0)
                diversity_penalty = 0.0
                for prev in selected:
                    diversity_penalty = max(diversity_penalty, self._candidate_similarity(cand, prev))
                    if str((cand.get("video") or {}).get("channel_title") or "").strip().lower() == str(
                        (prev.get("video") or {}).get("channel_title") or ""
                    ).strip().lower():
                        diversity_penalty = max(diversity_penalty, 0.24)
                mmr = 0.78 * relevance - 0.22 * diversity_penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    def _candidate_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        lv = left.get("video") or {}
        rv = right.get("video") or {}
        title_sim = self._token_jaccard(
            self._normalize_title_for_similarity(str(lv.get("title") or "")),
            self._normalize_title_for_similarity(str(rv.get("title") or "")),
        )
        strategy_left = str((left.get("query_candidate") or QueryCandidate("", "literal", 0.5)).strategy or "")
        strategy_right = str((right.get("query_candidate") or QueryCandidate("", "literal", 0.5)).strategy or "")
        strategy_bonus = 0.08 if strategy_left == strategy_right else 0.0
        return float(max(0.0, min(1.0, title_sim + strategy_bonus)))

    def _expand_controlled_synonyms(self, terms: list[str]) -> list[str]:
        expansions: list[str] = []
        seen: set[str] = set()
        synonym_map: dict[str, list[str]] = {
            "ai": ["artificial intelligence", "machine intelligence", "automated systems"],
            "hospital": ["medical center", "clinic", "healthcare facility"],
            "car crash": ["car accident", "vehicle collision", "dashcam crash"],
            "climate change": ["global warming", "wildfire smoke", "flooding street", "drought footage"],
            "economy": ["financial markets", "inflation", "cost of living"],
            "burnout": ["work stress", "overwork fatigue", "mental exhaustion"],
        }
        for raw in terms:
            value = " ".join(str(raw or "").lower().split()).strip()
            if not value:
                continue
            candidates = [value]
            for key, mapped in synonym_map.items():
                if key in value:
                    candidates.extend(mapped)
            if value.endswith("s"):
                candidates.append(value[:-1])
            else:
                candidates.append(f"{value}s")
            for token in value.split():
                if len(token) <= 4:
                    continue
                candidates.append(token)

            for item in candidates:
                cleaned = " ".join(item.split()).strip()
                if not cleaned:
                    continue
                if cleaned in seen:
                    continue
                seen.add(cleaned)
                expansions.append(cleaned)
        return expansions[:40]

    def _decompose_concept_for_recovery(self, title: str, summary: str, keywords: list[str]) -> list[str]:
        lowered = f"{title} {summary}".lower()
        decomposed = []
        if any(token in lowered for token in {"anxiety", "isolation", "burnout", "stress", "hope", "failure"}):
            decomposed.extend(
                [
                    "person worried at desk bills",
                    "close up hands typing late night",
                    "office at night tired expression",
                    "stock market red screen",
                    "empty wallet close up",
                ]
            )
        decomposed.extend(keywords[:4])
        decomposed.append(title)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in decomposed:
            clean = " ".join(str(item or "").split()).strip().lower()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped[:20]

    def _recover_candidates_from_local_corpus(
        self,
        conn,
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        visual_spec: dict[str, list[str]],
        preferred_video_duration: str,
        fast_mode: bool,
        existing_video_counts: dict[str, int],
        generated_video_counts: dict[str, int],
        max_segments_per_video: int,
        concept_title: str,
    ) -> list[dict[str, Any]]:
        rows = fetch_all(
            conn,
            """
            SELECT id, title, channel_title, description, duration_sec, COALESCE(view_count, 0) AS view_count, is_creative_commons, created_at
            FROM videos
            ORDER BY created_at DESC
            LIMIT 320
            """,
        )
        if not rows:
            return []

        query_candidate = QueryCandidate(
            text=f"local cache {concept_title}".strip(),
            strategy="recovery_adjacent",
            confidence=0.45,
            source_terms=[concept_title],
            weight=0.55,
            stage="recovery",
            source_surface="local_cache",
        )

        candidates: list[dict[str, Any]] = []
        for row in rows:
            video_id = str(row.get("id") or "").strip()
            if not video_id:
                continue
            existing_for_video = existing_video_counts.get(video_id, 0)
            generated_for_video = generated_video_counts.get(video_id, 0)
            if existing_for_video + generated_for_video >= max_segments_per_video:
                continue
            video = {
                "id": video_id,
                "title": str(row.get("title") or ""),
                "channel_title": str(row.get("channel_title") or ""),
                "description": str(row.get("description") or ""),
                "duration_sec": int(row.get("duration_sec") or 0),
                "view_count": int(row.get("view_count") or 0),
                "is_creative_commons": bool(row.get("is_creative_commons")),
                "published_at": "",
                "search_source": "local_cache",
                "query_strategy": "recovery_adjacent",
                "query_stage": "recovery",
                "search_query": "local_cache",
            }
            ranking = self._score_video_candidate(
                conn,
                video=video,
                query_candidate=query_candidate,
                concept_terms=concept_terms,
                context_terms=context_terms,
                concept_embedding=concept_embedding,
                subject_tag=subject_tag,
                visual_spec=visual_spec,
                preferred_video_duration=preferred_video_duration,
                stage_name="recovery",
                require_context=False,
                fast_mode=fast_mode,
            )
            discovery = float(ranking.get("discovery_score") or 0.0)
            text_score = float((ranking.get("text_relevance") or {}).get("score") or 0.0)
            if discovery < 0.11 and text_score < 0.08:
                continue
            candidates.append(
                {
                    "video": video,
                    "video_id": video_id,
                    "video_duration": int(video.get("duration_sec") or 0),
                    "video_relevance": ranking.get("text_relevance") or {},
                    "ranking": ranking,
                    "query_candidate": query_candidate,
                    "stage": "recovery",
                }
            )

        candidates.sort(key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0), reverse=True)
        return candidates[: (18 if fast_mode else 36)]

    def _build_query_variants(
        self,
        title: str,
        keywords: list[str],
        subject_tag: str | None,
        context_terms: list[str] | None = None,
    ) -> list[str]:
        short_keywords = [k.strip() for k in keywords if k.strip()][:3]
        disambiguators = [term for term in (context_terms or []) if term.strip()][:3]
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
            f"{core} {keyword_hint} crash course",
            f"{core} {keyword_hint} deep dive",
            f"{core} {keyword_hint} shorts",
            f"{core} {keyword_hint} short explanation",
            f"{core} {keyword_hint} quick tutorial",
            f"{core} {keyword_hint} complete class",
            f"{core} {keyword_hint} full tutorial",
            f"{core} {context_hint} practice problems",
            f"{core} {keyword_hint} concept review",
            f"{core} {context_hint} exam prep",
            f"{core} {keyword_hint} foundations",
            f"{core} explained",
            f"{core} {keyword_hint}",
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
            if len(deduped) >= 18:
                break
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
                "view_count": int(video.get("view_count") or 0),
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
        target_clip_duration_sec: int,
    ) -> SegmentMatch | None:
        title = str(video.get("title") or "").strip()
        description = str(video.get("description") or "").strip()
        metadata_text = " ".join(part for part in [title, description] if part).strip()
        if not metadata_text:
            return None

        duration_sec = int(video.get("duration_sec") or 0)
        desired_clip_len = self._normalize_target_clip_duration(target_clip_duration_sec)

        if duration_sec > 0:
            clip_len = min(desired_clip_len, max(15, duration_sec))
            if duration_sec <= clip_len:
                start = 0.0
            else:
                # Skip very early intro content when we must fall back to metadata-only clipping.
                start = max(0.0, min(float(duration_sec - clip_len), float(duration_sec) * 0.18))
            end = min(float(duration_sec), start + float(clip_len))
        else:
            start = 0.0
            end = float(desired_clip_len)

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

    def _build_dry_run_reel_preview(
        self,
        concept: dict[str, Any],
        video: dict[str, Any],
        segment: SegmentMatch,
        clip_window: tuple[int, int],
        relevance_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start_sec, end_sec = clip_window
        video_id = str(video.get("id") or "").strip()
        video_url = f"https://www.youtube.com/embed/{video_id}?start={start_sec}&end={end_sec}" if video_id else ""
        matched_terms = [
            str(term).strip()
            for term in (relevance_context or {}).get("matched_terms", [])
            if str(term).strip()
        ][:8]
        return {
            "reel_id": f"dry-run-{video_id}-{start_sec}-{end_sec}",
            "concept_id": str(concept.get("id") or ""),
            "concept_title": str(concept.get("title") or ""),
            "video_title": str(video.get("title") or "").strip(),
            "video_description": self._clean_video_description(str(video.get("description") or "")),
            "ai_summary": "",
            "video_url": video_url,
            "t_start": float(start_sec),
            "t_end": float(end_sec),
            "transcript_snippet": str(segment.text or "")[:700],
            "takeaways": [],
            "captions": [],
            "score": float(segment.score),
            "relevance_score": float((relevance_context or {}).get("score") or segment.score),
            "discovery_score": float((relevance_context or {}).get("discovery_score") or 0.0),
            "clipability_score": float((relevance_context or {}).get("clipability_score") or 0.0),
            "query_strategy": str((relevance_context or {}).get("query_strategy") or ""),
            "retrieval_stage": str((relevance_context or {}).get("retrieval_stage") or ""),
            "source_surface": str((relevance_context or {}).get("source_surface") or ""),
            "matched_terms": matched_terms,
            "relevance_reason": str((relevance_context or {}).get("reason") or "").strip(),
            "video_duration_sec": int(video.get("duration_sec") or 0),
            "clip_duration_sec": float(max(0.0, end_sec - start_sec)),
        }

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
        target_clip_duration_sec: int = DEFAULT_TARGET_CLIP_DURATION_SEC,
        target_clip_duration_min_sec: int | None = None,
        target_clip_duration_max_sec: int | None = None,
    ) -> dict[str, Any] | None:
        reel_id = str(uuid.uuid4())
        clip_min_len, clip_max_len, _ = self._resolve_clip_duration_bounds(
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        if clip_window is None:
            clip_window = self._normalize_clip_window(
                segment.t_start,
                segment.t_end,
                int(video.get("duration_sec") or 0),
                min_len=clip_min_len,
                max_len=clip_max_len,
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
            "discovery_score": float((relevance_context or {}).get("discovery_score") or 0.0),
            "clipability_score": float((relevance_context or {}).get("clipability_score") or 0.0),
            "query_strategy": str((relevance_context or {}).get("query_strategy") or ""),
            "retrieval_stage": str((relevance_context or {}).get("retrieval_stage") or ""),
            "source_surface": str((relevance_context or {}).get("source_surface") or ""),
            "matched_terms": matched_terms,
            "relevance_reason": relevance_reason,
            "video_duration_sec": int(video.get("duration_sec") or 0),
            "clip_duration_sec": float(max(0.0, end_sec - start_sec)),
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

    def _should_use_full_short_clip(
        self,
        prefer_short_query: bool,
        video_duration_sec: int,
        clip_min_len: int,
        clip_max_len: int,
    ) -> bool:
        if not prefer_short_query:
            return False
        if video_duration_sec <= 0:
            return False
        if clip_max_len > 0 and video_duration_sec > clip_max_len:
            return False
        if clip_min_len > 0 and video_duration_sec < max(8, int(round(clip_min_len * 0.55))):
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
        min_len: int = 15,
        max_len: int = 60,
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
                min_len=min_len,
                max_len=max_len,
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
        min_end = refined_start + float(min_len)
        max_end = refined_start + float(max_len)

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
            min_len=min_len,
            max_len=max_len,
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
                COALESCE(v.duration_sec, 0) AS video_duration_sec,
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
                    "discovery_score": float(relevance_context.get("score") or 0.0),
                    "clipability_score": float(self._score_clipability_from_metadata({"duration_sec": row.get("video_duration_sec")}, strategy="literal")),
                    "query_strategy": "",
                    "retrieval_stage": "",
                    "source_surface": "",
                    "matched_terms": relevance_context.get("matched_terms", []),
                    "relevance_reason": str(relevance_context.get("reason") or ""),
                    "concept_position": concept_order.get(row["concept_id"]),
                    "total_concepts": total_concepts,
                    "video_duration_sec": int(row.get("video_duration_sec") or 0),
                    "clip_duration_sec": round(max(0.0, float(row["t_end"]) - float(row["t_start"])), 2),
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
