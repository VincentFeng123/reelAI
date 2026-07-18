import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import get_settings

logger = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg import errors as pg_errors
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency at runtime
    psycopg = None
    pg_errors = None
    dict_row = None


class DatabaseIntegrityError(Exception):
    pass


LEGACY_LEARNER_ID = "legacy"


# When init_db() runs during container startup on Railway, the Postgres add-on
# may still be in its own boot sequence — ``psycopg.connect`` then raises
# ``OperationalError: the database system is starting up``. We retry with
# exponential backoff rather than let the FastAPI lifespan crash.
INIT_DB_MAX_RETRIES = 10
INIT_DB_INITIAL_BACKOFF_SEC = 1.0
INIT_DB_MAX_BACKOFF_SEC = 15.0


def _is_transient_postgres_startup_error(exc: BaseException) -> bool:
    """True when the error text indicates PG is still booting or unreachable.

    Matches the wording of both ``FATAL: the database system is starting up``
    and common ``Connection refused`` / ``could not translate host name``
    transport errors that clear within a few seconds of container start.
    """
    text = str(exc).lower()
    transient_markers = (
        "the database system is starting up",
        "the database system is shutting down",
        "connection refused",
        "could not translate host name",
        "temporary failure in name resolution",
        "no route to host",
        "connection to server",
        "connection timed out",
    )
    return any(marker in text for marker in transient_markers)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS materials (
    id TEXT PRIMARY KEY,
    subject_tag TEXT,
    knowledge_level TEXT NOT NULL DEFAULT 'beginner',
    level_adjustment REAL NOT NULL DEFAULT 0.0,
    raw_text TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    title TEXT NOT NULL,
    keywords_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_concepts_material_id ON concepts(material_id);

CREATE TABLE IF NOT EXISTS material_chunks (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_material_chunks_material_id ON material_chunks(material_id);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    channel_title TEXT,
    channel_id TEXT,
    description TEXT,
    duration_sec INTEGER,
    view_count INTEGER DEFAULT 0,
    published_at TEXT,
    source_url TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    is_creative_commons INTEGER DEFAULT 0,
    provider TEXT DEFAULT 'youtube',
    playback_url TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    text TEXT NOT NULL,
    embedding_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_transcript_chunks_video_id ON transcript_chunks(video_id);

CREATE TABLE IF NOT EXISTS reel_generations (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    concept_id TEXT,
    request_key TEXT NOT NULL,
    generation_mode TEXT NOT NULL DEFAULT 'slow',
    retrieval_profile TEXT NOT NULL DEFAULT 'bootstrap',
    status TEXT NOT NULL DEFAULT 'pending',
    source_generation_id TEXT,
    reel_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    activated_at TEXT,
    error_text TEXT,
    lesson_order_json TEXT,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_generations_material_request_created
ON reel_generations(material_id, request_key, created_at DESC);

CREATE TABLE IF NOT EXISTS reel_generation_heads (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    request_key TEXT NOT NULL,
    active_generation_id TEXT NOT NULL,
    refinement_state_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_generation_heads_active
ON reel_generation_heads(active_generation_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_generation_heads_material_request
ON reel_generation_heads(material_id, request_key);

CREATE TABLE IF NOT EXISTS reel_generation_jobs (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    concept_id TEXT,
    request_key TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL DEFAULT '',
    learner_id TEXT NOT NULL DEFAULT 'legacy',
    source_generation_id TEXT NOT NULL DEFAULT '',
    result_generation_id TEXT,
    target_profile TEXT NOT NULL DEFAULT 'deep',
    request_params_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    phase TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0.0,
    lease_owner TEXT,
    lease_expires_at TEXT,
    heartbeat_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 2,
    deadline_at TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    cancel_requested_at TEXT,
    model_used TEXT,
    quality_degraded INTEGER NOT NULL DEFAULT 0,
    usage_json TEXT NOT NULL DEFAULT '{}',
    terminal_error_code TEXT,
    terminal_error_message TEXT,
    terminal_error_json TEXT,
    next_event_seq INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    started_at TEXT,
    completed_at TEXT,
    error_text TEXT,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_generation_jobs_material_request_status
ON reel_generation_jobs(material_id, request_key, status, created_at DESC);

CREATE TABLE IF NOT EXISTS generation_job_events (
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY(job_id, seq),
    FOREIGN KEY(job_id) REFERENCES reel_generation_jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_generation_job_events_job_seq
ON generation_job_events(job_id, seq);

CREATE TABLE IF NOT EXISTS api_idempotency_records (
    scope TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',
    resource_id TEXT NOT NULL,
    attempt_token TEXT NOT NULL DEFAULT '',
    response_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(scope, learner_id, key_hash)
);

CREATE INDEX IF NOT EXISTS idx_api_idempotency_records_updated
ON api_idempotency_records(status, updated_at);

CREATE INDEX IF NOT EXISTS idx_api_idempotency_records_resource
ON api_idempotency_records(scope, resource_id);

CREATE TABLE IF NOT EXISTS reels (
    id TEXT PRIMARY KEY,
    generation_id TEXT,
    material_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    video_url TEXT NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    transcript_snippet TEXT NOT NULL,
    takeaways_json TEXT NOT NULL,
    ai_summary TEXT NOT NULL DEFAULT '',
    match_reason TEXT NOT NULL DEFAULT '',
    informativeness REAL,
    base_score REAL NOT NULL,
    difficulty REAL,
    model_used TEXT,
    quality_degraded INTEGER NOT NULL DEFAULT 0,
    duration_preference_met INTEGER,
    duration_fit TEXT,
    selected_cue_ids_json TEXT NOT NULL DEFAULT '[]',
    search_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_reels_material_id ON reels(material_id);
CREATE INDEX IF NOT EXISTS idx_reels_concept_id ON reels(concept_id);

CREATE TABLE IF NOT EXISTS reel_feedback (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL DEFAULT 'legacy',
    reel_id TEXT NOT NULL,
    helpful INTEGER NOT NULL DEFAULT 0,
    confusing INTEGER NOT NULL DEFAULT 0,
    rating INTEGER,
    saved INTEGER NOT NULL DEFAULT 0,
    mastery_updated_at TEXT,
    updated_at TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(reel_id) REFERENCES reels(id)
);

CREATE INDEX IF NOT EXISTS idx_reel_feedback_reel_id ON reel_feedback(reel_id);

CREATE TABLE IF NOT EXISTS learner_material_progress (
    learner_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    selected_level TEXT NOT NULL DEFAULT 'beginner',
    global_adjustment REAL NOT NULL DEFAULT 0.0,
    difficulty_reset_at TEXT NOT NULL,
    feedback_revision INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (learner_id, material_id),
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_learner_material_progress_material
ON learner_material_progress(material_id);

CREATE TABLE IF NOT EXISTS learner_reel_progress (
    learner_id TEXT NOT NULL,
    reel_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    max_fraction REAL NOT NULL DEFAULT 0.0,
    scrolled_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (learner_id, reel_id),
    FOREIGN KEY(reel_id) REFERENCES reels(id),
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_learner_reel_progress_material_completed
ON learner_reel_progress(learner_id, material_id, completed_at);

CREATE TABLE IF NOT EXISTS reel_assessment_questions (
    id TEXT PRIMARY KEY,
    reel_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    prompt TEXT NOT NULL,
    options_json TEXT NOT NULL,
    correct_index INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(reel_id) REFERENCES reels(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_assessment_questions_reel_fingerprint
ON reel_assessment_questions(reel_id, fingerprint);

CREATE INDEX IF NOT EXISTS idx_reel_assessment_questions_reel
ON reel_assessment_questions(reel_id);

CREATE TABLE IF NOT EXISTS assessment_sessions (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    current_index INTEGER NOT NULL DEFAULT 0,
    question_count INTEGER NOT NULL DEFAULT 0,
    correct_count INTEGER NOT NULL DEFAULT 0,
    information_units REAL NOT NULL DEFAULT 0.0,
    readiness_threshold REAL NOT NULL DEFAULT 3.5,
    organizer_checkpoint_reel_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    snoozed_at TEXT,
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_assessment_sessions_one_pending
ON assessment_sessions(learner_id, material_id)
WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_assessment_sessions_history
ON assessment_sessions(learner_id, material_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS assessment_session_questions (
    session_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (session_id, question_id),
    FOREIGN KEY(session_id) REFERENCES assessment_sessions(id),
    FOREIGN KEY(question_id) REFERENCES reel_assessment_questions(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_assessment_session_questions_position
ON assessment_session_questions(session_id, position);

CREATE TABLE IF NOT EXISTS assessment_attempts (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    choice_index INTEGER NOT NULL,
    is_correct INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES assessment_sessions(id),
    FOREIGN KEY(question_id) REFERENCES reel_assessment_questions(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_assessment_attempts_session_question
ON assessment_attempts(session_id, question_id);

CREATE INDEX IF NOT EXISTS idx_assessment_attempts_learner_question
ON assessment_attempts(learner_id, question_id);

CREATE TABLE IF NOT EXISTS assessment_concept_outcomes (
    learner_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    correct_count INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    adjustment REAL NOT NULL DEFAULT 0.0,
    source_reel_id TEXT,
    source_video_id TEXT,
    source_difficulty REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, concept_id),
    FOREIGN KEY(session_id) REFERENCES assessment_sessions(id),
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_assessment_concept_outcomes_learner_material
ON assessment_concept_outcomes(learner_id, material_id, created_at DESC);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    concept_title TEXT NOT NULL,
    selected_video_id TEXT,
    failure_reason TEXT NOT NULL DEFAULT '',
    debug_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(material_id) REFERENCES materials(id),
    FOREIGN KEY(concept_id) REFERENCES concepts(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_material_concept ON retrieval_runs(material_id, concept_id, created_at DESC);

CREATE TABLE IF NOT EXISTS retrieval_queries (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    strategy TEXT NOT NULL,
    stage TEXT NOT NULL,
    source_surface TEXT NOT NULL DEFAULT '',
    source_terms_json TEXT NOT NULL DEFAULT '[]',
    weight REAL NOT NULL DEFAULT 0.0,
    result_count INTEGER NOT NULL DEFAULT 0,
    kept_count INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES retrieval_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_queries_run_stage ON retrieval_queries(run_id, stage, strategy);

CREATE TABLE IF NOT EXISTS retrieval_candidates (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    video_title TEXT NOT NULL DEFAULT '',
    channel_title TEXT NOT NULL DEFAULT '',
    strategy TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT '',
    query_text TEXT NOT NULL DEFAULT '',
    source_surface TEXT NOT NULL DEFAULT '',
    discovery_score REAL NOT NULL DEFAULT 0.0,
    clipability_score REAL NOT NULL DEFAULT 0.0,
    final_score REAL NOT NULL DEFAULT 0.0,
    feature_json TEXT NOT NULL DEFAULT '{}',
    position INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES retrieval_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_run_position ON retrieval_candidates(run_id, position);

CREATE TABLE IF NOT EXISTS retrieval_outcomes (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    t_start REAL NOT NULL DEFAULT 0,
    t_end REAL NOT NULL DEFAULT 0,
    reason_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES retrieval_runs(id),
    FOREIGN KEY(video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_outcomes_run ON retrieval_outcomes(run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS community_accounts (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    username_normalized TEXT NOT NULL UNIQUE,
    email TEXT,
    email_normalized TEXT,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    verified_at TEXT,
    verification_code_hash TEXT,
    verification_expires_at TEXT,
    legacy_claim_owner_key_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS community_signup_email_verifications (
    id TEXT PRIMARY KEY,
    owner_key_hash TEXT NOT NULL,
    email TEXT NOT NULL,
    email_normalized TEXT NOT NULL,
    verified_at TEXT,
    verification_code_hash TEXT,
    verification_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_community_signup_email_verifications_owner_email
ON community_signup_email_verifications(owner_key_hash, email_normalized);

CREATE INDEX IF NOT EXISTS idx_community_signup_email_verifications_email_updated_at
ON community_signup_email_verifications(email_normalized, updated_at DESC);

CREATE TABLE IF NOT EXISTS community_sessions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY(account_id) REFERENCES community_accounts(id)
);

CREATE INDEX IF NOT EXISTS idx_community_sessions_account_id ON community_sessions(account_id);

CREATE TABLE IF NOT EXISTS community_sets (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    reels_json TEXT NOT NULL,
    reel_count INTEGER NOT NULL DEFAULT 0,
    curator TEXT NOT NULL,
    likes INTEGER NOT NULL DEFAULT 0,
    dislikes INTEGER NOT NULL DEFAULT 0,
    learners INTEGER NOT NULL DEFAULT 1,
    updated_label TEXT NOT NULL,
    thumbnail_url TEXT NOT NULL,
    owner_key_hash TEXT,
    owner_account_id TEXT,
    visibility TEXT NOT NULL DEFAULT 'public',
    featured INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_community_sets_featured_created_at ON community_sets(featured DESC, created_at DESC);

-- Per-account vote on a community set. `vote` is 'like' or 'dislike';
-- mutual exclusion is enforced at write time (there is at most one row
-- per (account_id, set_id), and toggling a vote rewrites it). Aggregate
-- counts on `community_sets.likes` / `community_sets.dislikes` are
-- recomputed from this table each time a vote is recorded so the
-- numbers can never drift.
CREATE TABLE IF NOT EXISTS community_set_votes (
    account_id TEXT NOT NULL,
    set_id TEXT NOT NULL,
    vote TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (account_id, set_id)
);

CREATE INDEX IF NOT EXISTS idx_community_set_votes_set_id ON community_set_votes(set_id);

CREATE TABLE IF NOT EXISTS community_material_history (
    account_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    title TEXT NOT NULL,
    updated_at BIGINT NOT NULL,
    starred INTEGER NOT NULL DEFAULT 0,
    generation_mode TEXT NOT NULL DEFAULT 'slow',
    source TEXT NOT NULL DEFAULT 'search',
    feed_query TEXT,
    active_index INTEGER,
    active_reel_id TEXT,
    recall_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(account_id, material_id),
    FOREIGN KEY(account_id) REFERENCES community_accounts(id)
);

CREATE INDEX IF NOT EXISTS idx_community_material_history_account_updated_at
ON community_material_history(account_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS community_account_settings (
    account_id TEXT PRIMARY KEY,
    generation_mode TEXT NOT NULL DEFAULT 'slow',
    default_input_mode TEXT NOT NULL DEFAULT 'source',
    min_relevance_threshold REAL NOT NULL DEFAULT 0.3,
    start_muted INTEGER NOT NULL DEFAULT 1,
    creative_commons_only INTEGER NOT NULL DEFAULT 0,
    preferred_video_duration TEXT NOT NULL DEFAULT 'any',
    target_clip_duration_sec INTEGER NOT NULL DEFAULT 55,
    target_clip_duration_min_sec INTEGER NOT NULL DEFAULT 20,
    target_clip_duration_max_sec INTEGER NOT NULL DEFAULT 55,
    autoplay_next_reel INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(account_id) REFERENCES community_accounts(id)
);

CREATE TABLE IF NOT EXISTS billing_provider_customers (
    account_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_environment TEXT NOT NULL DEFAULT 'Unknown',
    external_customer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(account_id, provider, provider_environment),
    UNIQUE(provider, provider_environment, external_customer_id),
    FOREIGN KEY(account_id) REFERENCES community_accounts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS billing_subscriptions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_environment TEXT NOT NULL DEFAULT 'Unknown',
    external_subscription_id TEXT NOT NULL,
    external_product_id TEXT NOT NULL,
    plan_code TEXT NOT NULL,
    status TEXT NOT NULL,
    current_period_end TEXT,
    cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
    provider_event_created_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(provider, external_subscription_id),
    FOREIGN KEY(account_id) REFERENCES community_accounts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_billing_subscriptions_account_status_end
ON billing_subscriptions(account_id, status, current_period_end);

CREATE TABLE IF NOT EXISTS billing_provider_events (
    provider TEXT NOT NULL,
    external_event_id TEXT NOT NULL,
    external_event_created_at TEXT,
    event_type TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY(provider, external_event_id)
);

CREATE TABLE IF NOT EXISTS daily_search_usage (
    account_id TEXT NOT NULL,
    usage_day TEXT NOT NULL,
    used_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(account_id, usage_day),
    FOREIGN KEY(account_id) REFERENCES community_accounts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_quota_reservations (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    usage_day TEXT NOT NULL,
    surface TEXT NOT NULL,
    plan_code TEXT NOT NULL DEFAULT 'free',
    material_id TEXT,
    generation_job_id TEXT,
    status TEXT NOT NULL DEFAULT 'reserved',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finalized_at TEXT,
    UNIQUE(account_id, operation_key),
    FOREIGN KEY(account_id) REFERENCES community_accounts(id) ON DELETE CASCADE,
    FOREIGN KEY(generation_job_id) REFERENCES reel_generation_jobs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_search_quota_reservations_day_status
ON search_quota_reservations(account_id, usage_day, status);

CREATE INDEX IF NOT EXISTS idx_search_quota_reservations_generation_job
ON search_quota_reservations(generation_job_id);

CREATE TABLE IF NOT EXISTS search_quota_reservation_jobs (
    reservation_id TEXT NOT NULL,
    generation_job_id TEXT NOT NULL,
    attached_at TEXT NOT NULL,
    PRIMARY KEY(reservation_id, generation_job_id),
    FOREIGN KEY(reservation_id) REFERENCES search_quota_reservations(id) ON DELETE CASCADE,
    FOREIGN KEY(generation_job_id) REFERENCES reel_generation_jobs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_search_quota_reservation_jobs_generation
ON search_quota_reservation_jobs(generation_job_id);

CREATE TABLE IF NOT EXISTS rate_limit_events (
    id TEXT PRIMARY KEY,
    rate_key TEXT NOT NULL,
    hit_at REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_events_key_hit_at ON rate_limit_events(rate_key, hit_at);
CREATE INDEX IF NOT EXISTS idx_rate_limit_events_hit_at ON rate_limit_events(hit_at);

CREATE TABLE IF NOT EXISTS search_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS supadata_search_cache (
    cache_key TEXT PRIMARY KEY,
    normalized_query TEXT NOT NULL,
    filters_json TEXT NOT NULL DEFAULT '{}',
    language TEXT NOT NULL DEFAULT 'en',
    page_token TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT 'supadata',
    schema_version TEXT NOT NULL,
    response_json TEXT NOT NULL,
    result_count INTEGER NOT NULL DEFAULT 0,
    is_empty INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_supadata_search_cache_expires
ON supadata_search_cache(expires_at);

CREATE TABLE IF NOT EXISTS transcript_cache (
    video_id TEXT PRIMARY KEY,
    transcript_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_artifacts (
    cache_key TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    requested_language TEXT NOT NULL DEFAULT 'en',
    returned_language TEXT NOT NULL DEFAULT '',
    native_mode INTEGER NOT NULL DEFAULT 1,
    schema_version TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    duration_sec REAL NOT NULL DEFAULT 0.0,
    cue_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transcript_artifacts_video_lookup
ON transcript_artifacts(video_id, provider, requested_language, native_mode, schema_version, expires_at);

CREATE TABLE IF NOT EXISTS generation_provider_usage (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    operation TEXT NOT NULL,
    model TEXT,
    billable_requests INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES reel_generation_jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_generation_provider_usage_job_created
ON generation_provider_usage(job_id, created_at);

CREATE TABLE IF NOT EXISTS concept_search_terms (
    id TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    term TEXT NOT NULL,
    term_kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(concept_id, term, term_kind),
    FOREIGN KEY(concept_id) REFERENCES concepts(id),
    FOREIGN KEY(material_id) REFERENCES materials(id)
);

CREATE INDEX IF NOT EXISTS idx_concept_search_terms_material_concept
ON concept_search_terms(material_id, concept_id, created_at);

CREATE TABLE IF NOT EXISTS blocked_video_tombstones (
    video_id TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ranked_feed_cache (
    cache_key TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    generation_id TEXT NOT NULL DEFAULT '',
    fast_mode INTEGER NOT NULL DEFAULT 0,
    source_fingerprint TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ranked_feed_cache_material_generation
ON ranked_feed_cache(material_id, generation_id, fast_mode);

CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT PRIMARY KEY,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_liveness_cache (
    video_id TEXT PRIMARY KEY,
    alive INTEGER NOT NULL,
    checked_at TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL DEFAULT 28800
);

CREATE TABLE IF NOT EXISTS community_starred_sets (
    account_id TEXT NOT NULL,
    set_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (account_id, set_id)
);

CREATE TABLE IF NOT EXISTS community_feed_snapshots (
    account_id TEXT NOT NULL,
    material_key TEXT NOT NULL,
    snapshot_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (account_id, material_key)
);

CREATE TABLE IF NOT EXISTS community_drafts (
    account_id TEXT NOT NULL,
    draft_key TEXT NOT NULL,
    draft_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (account_id, draft_key)
);

CREATE TABLE IF NOT EXISTS community_material_seeds (
    account_id TEXT NOT NULL,
    material_id TEXT NOT NULL,
    seed_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (account_id, material_id)
);

CREATE TABLE IF NOT EXISTS community_material_groups (
    account_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    group_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (account_id, group_id)
);
"""

_db_init_lock = threading.Lock()
_db_ready = False
DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 120000
_SAFE_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LEGACY_COMMUNITY_OWNER_HASH = "__legacy_unowned__"
DEFAULT_COMMUNITY_VISIBILITY = "public"
PUBLIC_COMMUNITY_VISIBILITY = "public"


def _db_path() -> str:
    settings = get_settings()
    os.makedirs(settings.data_dir, exist_ok=True)
    return os.path.join(settings.data_dir, "studyreels.db")


def _sqlite_busy_timeout_ms() -> int:
    raw = os.getenv("SQLITE_BUSY_TIMEOUT_MS", "").strip()
    if not raw:
        return DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    return max(1000, min(600000, parsed))


def _database_url() -> str:
    settings = get_settings()
    raw = (settings.database_url or "").strip() or os.getenv("DATABASE_URL", "").strip()
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://") :]
    return raw


def _is_postgres_configured() -> bool:
    url = _database_url()
    return url.startswith("postgresql://")


def _is_postgres_conn(conn: Any) -> bool:
    if psycopg is None:
        return False
    return conn.__class__.__module__.startswith("psycopg")


def _adapt_query_for_postgres(query: str) -> str:
    result: list[str] = []
    i = 0
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    while i < len(query):
        char = query[i]
        nxt = query[i + 1] if i + 1 < len(query) else ""

        if char == "%":
            if nxt == "%":
                result.extend((char, nxt))
                i += 2
                continue
            if nxt in {"s", "b", "t"}:
                result.append(char)
                i += 1
                continue
            result.append("%%")
            i += 1
            continue

        if in_line_comment:
            result.append(char)
            if char == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            result.append(char)
            if char == "*" and nxt == "/":
                result.append(nxt)
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if not in_single_quote and not in_double_quote:
            if char == "-" and nxt == "-":
                result.append(char)
                result.append(nxt)
                in_line_comment = True
                i += 2
                continue
            if char == "/" and nxt == "*":
                result.append(char)
                result.append(nxt)
                in_block_comment = True
                i += 2
                continue
            if char == "?":
                result.append("%s")
                i += 1
                continue

        result.append(char)
        if char == "'" and not in_double_quote:
            if in_single_quote and nxt == "'":
                result.append(nxt)
                i += 2
                continue
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        i += 1
    return "".join(result)


def _strip_sql_line_comments(statement: str) -> str:
    lines = [line for line in statement.splitlines() if not line.strip().startswith("--")]
    return "\n".join(lines).strip()


def _postgres_schema_statements() -> list[str]:
    statements: list[str] = []
    for chunk in SCHEMA.split(";"):
        stmt = _strip_sql_line_comments(chunk)
        if not stmt:
            continue
        upper = stmt.upper()
        if upper.startswith("PRAGMA "):
            continue
        if re.search(r"\bROWID\b", upper):
            continue
        statements.append(stmt)
    return statements


def _migrate_reel_feedback_uniqueness_sqlite(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(reel_feedback)").fetchall()}
    if "learner_id" not in columns:
        conn.execute(
            f"ALTER TABLE reel_feedback ADD COLUMN learner_id TEXT NOT NULL DEFAULT '{LEGACY_LEARNER_ID}'"
        )
    if "mastery_updated_at" not in columns:
        conn.execute("ALTER TABLE reel_feedback ADD COLUMN mastery_updated_at TEXT")
    if "updated_at" not in columns:
        conn.execute("ALTER TABLE reel_feedback ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
    conn.execute(
        "UPDATE reel_feedback SET learner_id = ? WHERE learner_id IS NULL OR TRIM(learner_id) = ''",
        (LEGACY_LEARNER_ID,),
    )
    conn.execute(
        "UPDATE reel_feedback SET updated_at = created_at WHERE updated_at IS NULL OR TRIM(updated_at) = ''"
    )
    conn.execute(
        """
        UPDATE reel_feedback
        SET mastery_updated_at = created_at
        WHERE mastery_updated_at IS NULL AND (helpful <> 0 OR confusing <> 0)
        """
    )
    conn.execute(
        """
        DELETE FROM reel_feedback
        WHERE EXISTS (
            SELECT 1
            FROM reel_feedback AS newer
            WHERE newer.learner_id = reel_feedback.learner_id
              AND newer.reel_id = reel_feedback.reel_id
              AND (
                  newer.updated_at > reel_feedback.updated_at
                  OR (newer.updated_at = reel_feedback.updated_at AND newer.rowid > reel_feedback.rowid)
              )
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_reel_feedback_reel_id")
    conn.execute("DROP INDEX IF EXISTS idx_reel_feedback_reel_id_unique")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reel_feedback_reel_id ON reel_feedback(reel_id)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_feedback_learner_reel_unique "
        "ON reel_feedback(learner_id, reel_id)"
    )


def _migrate_reel_feedback_uniqueness_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"ALTER TABLE reel_feedback ADD COLUMN IF NOT EXISTS learner_id TEXT NOT NULL DEFAULT '{LEGACY_LEARNER_ID}'"
        )
        cur.execute("ALTER TABLE reel_feedback ADD COLUMN IF NOT EXISTS mastery_updated_at TEXT")
        cur.execute("ALTER TABLE reel_feedback ADD COLUMN IF NOT EXISTS updated_at TEXT NOT NULL DEFAULT ''")
        cur.execute(
            "UPDATE reel_feedback SET learner_id = %s WHERE learner_id IS NULL OR BTRIM(learner_id) = ''",
            (LEGACY_LEARNER_ID,),
        )
        cur.execute(
            "UPDATE reel_feedback SET updated_at = created_at WHERE updated_at IS NULL OR BTRIM(updated_at) = ''"
        )
        cur.execute(
            """
            UPDATE reel_feedback
            SET mastery_updated_at = created_at
            WHERE mastery_updated_at IS NULL AND (helpful <> 0 OR confusing <> 0)
            """
        )
        cur.execute(
            """
            DELETE FROM reel_feedback AS older
            USING reel_feedback AS newer
            WHERE older.learner_id = newer.learner_id
              AND older.reel_id = newer.reel_id
              AND (
                  older.updated_at < newer.updated_at
                  OR (older.updated_at = newer.updated_at AND older.ctid < newer.ctid)
              )
            """
        )
        cur.execute("DROP INDEX IF EXISTS idx_reel_feedback_reel_id")
        cur.execute("DROP INDEX IF EXISTS idx_reel_feedback_reel_id_unique")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reel_feedback_reel_id ON reel_feedback(reel_id)")
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_feedback_learner_reel_unique "
            "ON reel_feedback(learner_id, reel_id)"
        )


def _migrate_reels_unique_clip_index_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        DELETE FROM reels
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM reels
            GROUP BY material_id, COALESCE(generation_id, ''), concept_id, video_id, t_start, t_end
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_reels_material_video_unique")
    conn.execute("DROP INDEX IF EXISTS idx_reels_material_video_clip_unique")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_reels_material_video_clip_unique
        ON reels(material_id, COALESCE(generation_id, ''), concept_id, video_id, t_start, t_end)
        """
    )


def _migrate_reels_unique_clip_index_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM reels AS older
            USING reels AS newer
            WHERE older.material_id = newer.material_id
              AND COALESCE(older.generation_id, '') = COALESCE(newer.generation_id, '')
              AND older.concept_id = newer.concept_id
              AND older.video_id = newer.video_id
              AND older.t_start = newer.t_start
              AND older.t_end = newer.t_end
              AND older.ctid < newer.ctid
            """
        )
        cur.execute("DROP INDEX IF EXISTS idx_reels_material_video_unique")
        cur.execute("DROP INDEX IF EXISTS idx_reels_material_video_clip_unique")
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_reels_material_video_clip_unique
            ON reels(material_id, COALESCE(generation_id, ''), concept_id, video_id, t_start, t_end)
            """
        )


def _migrate_knowledge_level_sqlite(conn: sqlite3.Connection) -> None:
    """Add knowledge-level columns to pre-existing DBs (sqlite lacks
    ADD COLUMN IF NOT EXISTS)."""
    material_cols = {r[1] for r in conn.execute("PRAGMA table_info(materials)").fetchall()}
    if "knowledge_level" not in material_cols:
        conn.execute(
            "ALTER TABLE materials ADD COLUMN knowledge_level TEXT NOT NULL DEFAULT 'beginner'"
        )
    if "level_adjustment" not in material_cols:
        conn.execute(
            "ALTER TABLE materials ADD COLUMN level_adjustment REAL NOT NULL DEFAULT 0.0"
        )
    reel_cols = {r[1] for r in conn.execute("PRAGMA table_info(reels)").fetchall()}
    if "difficulty" not in reel_cols:
        conn.execute("ALTER TABLE reels ADD COLUMN difficulty REAL")


def _migrate_reel_learning_content_sqlite(conn: sqlite3.Connection) -> None:
    """Add one-pass learning-content columns to pre-existing reel tables."""
    reel_cols = {r[1] for r in conn.execute("PRAGMA table_info(reels)").fetchall()}
    if "ai_summary" not in reel_cols:
        conn.execute("ALTER TABLE reels ADD COLUMN ai_summary TEXT NOT NULL DEFAULT ''")
    if "match_reason" not in reel_cols:
        conn.execute("ALTER TABLE reels ADD COLUMN match_reason TEXT NOT NULL DEFAULT ''")
    if "informativeness" not in reel_cols:
        conn.execute("ALTER TABLE reels ADD COLUMN informativeness REAL")


def _migrate_assessment_scroll_sqlite(conn: sqlite3.Connection) -> None:
    """Keep forward-scroll cadence separate from watch completion analytics."""
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(learner_reel_progress)").fetchall()
    }
    if "scrolled_at" not in columns:
        conn.execute("ALTER TABLE learner_reel_progress ADD COLUMN scrolled_at TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_learner_reel_progress_material_scrolled "
        "ON learner_reel_progress(learner_id, material_id, scrolled_at)"
    )


def _migrate_assessment_scroll_postgres(conn: Any) -> None:
    """Idempotently add the scroll-cadence field to deployed PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE learner_reel_progress "
            "ADD COLUMN IF NOT EXISTS scrolled_at TEXT"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_learner_reel_progress_material_scrolled "
            "ON learner_reel_progress(learner_id, material_id, scrolled_at)"
        )


_DURABLE_JOB_SQLITE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("content_fingerprint", "TEXT NOT NULL DEFAULT ''"),
    ("learner_id", f"TEXT NOT NULL DEFAULT '{LEGACY_LEARNER_ID}'"),
    ("phase", "TEXT NOT NULL DEFAULT 'queued'"),
    ("progress", "REAL NOT NULL DEFAULT 0.0"),
    ("lease_owner", "TEXT"),
    ("lease_expires_at", "TEXT"),
    ("heartbeat_at", "TEXT"),
    ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
    ("max_attempts", "INTEGER NOT NULL DEFAULT 2"),
    ("deadline_at", "TEXT"),
    ("cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
    ("cancel_requested_at", "TEXT"),
    ("model_used", "TEXT"),
    ("quality_degraded", "INTEGER NOT NULL DEFAULT 0"),
    ("usage_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("terminal_error_code", "TEXT"),
    ("terminal_error_message", "TEXT"),
    ("terminal_error_json", "TEXT"),
    ("next_event_seq", "INTEGER NOT NULL DEFAULT 0"),
    ("updated_at", "TEXT NOT NULL DEFAULT ''"),
)

_DURABLE_VIDEO_SQLITE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("channel_id", "TEXT"),
    ("published_at", "TEXT"),
    ("source_url", "TEXT"),
    ("metadata_json", "TEXT NOT NULL DEFAULT '{}'"),
)

_DURABLE_REEL_SQLITE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("model_used", "TEXT"),
    ("quality_degraded", "INTEGER NOT NULL DEFAULT 0"),
    ("duration_preference_met", "INTEGER"),
    ("duration_fit", "TEXT"),
    ("selected_cue_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("search_context_json", "TEXT NOT NULL DEFAULT '{}'"),
)


def _sqlite_add_missing_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[tuple[str, str], ...],
) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, declaration in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _migrate_lesson_order_sqlite(conn: sqlite3.Connection) -> None:
    """Idempotently add durable lesson ordering to legacy SQLite generations."""
    _sqlite_add_missing_columns(
        conn,
        "reel_generations",
        (("lesson_order_json", "TEXT"),),
    )


def _migrate_lesson_order_postgres(conn: Any) -> None:
    """Idempotently add durable lesson ordering to legacy PostgreSQL generations."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE reel_generations "
            "ADD COLUMN IF NOT EXISTS lesson_order_json TEXT"
        )


def _migrate_assessment_checkpoint_sqlite(conn: sqlite3.Connection) -> None:
    """Mark intentionally variable-size organizer checkpoint sessions."""
    _sqlite_add_missing_columns(
        conn,
        "assessment_sessions",
        (("organizer_checkpoint_reel_id", "TEXT"),),
    )


def _migrate_assessment_checkpoint_postgres(conn: Any) -> None:
    """Mark intentionally variable-size organizer checkpoint sessions."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE assessment_sessions "
            "ADD COLUMN IF NOT EXISTS organizer_checkpoint_reel_id TEXT"
        )


def _migrate_api_idempotency_sqlite(conn: sqlite3.Connection) -> None:
    """Add owner fencing to idempotency rows created by older builds."""
    _sqlite_add_missing_columns(
        conn,
        "api_idempotency_records",
        (("attempt_token", "TEXT NOT NULL DEFAULT ''"),),
    )


def _migrate_api_idempotency_postgres(conn: Any) -> None:
    """Add owner fencing to idempotency rows created by older builds."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE api_idempotency_records "
            "ADD COLUMN IF NOT EXISTS attempt_token TEXT NOT NULL DEFAULT ''"
        )


def _migrate_billing_subscription_environment_sqlite(
    conn: sqlite3.Connection,
) -> None:
    """Fail closed when upgrading rows created before environment isolation."""
    _sqlite_add_missing_columns(
        conn,
        "billing_subscriptions",
        (("provider_environment", "TEXT NOT NULL DEFAULT 'Unknown'"),),
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS "
        "idx_billing_subscriptions_account_environment_status_end "
        "ON billing_subscriptions("
        "account_id, provider_environment, status, current_period_end)"
    )


def _migrate_billing_provider_customer_environment_sqlite(
    conn: sqlite3.Connection,
) -> None:
    """Re-key legacy mappings so test and live Stripe IDs cannot collide."""
    table_info = conn.execute(
        "PRAGMA table_info(billing_provider_customers)"
    ).fetchall()
    columns = {str(row[1]) for row in table_info}
    primary_key = tuple(
        str(row[1])
        for row in sorted(table_info, key=lambda row: int(row[5] or 0))
        if int(row[5] or 0) > 0
    )
    unique_columns: set[tuple[str, ...]] = set()
    for index_row in conn.execute(
        "PRAGMA index_list(billing_provider_customers)"
    ).fetchall():
        if not int(index_row[2] or 0):
            continue
        index_name = str(index_row[1]).replace('"', '""')
        index_columns = tuple(
            str(row[2])
            for row in conn.execute(f'PRAGMA index_info("{index_name}")').fetchall()
        )
        unique_columns.add(index_columns)
    if (
        "provider_environment" in columns
        and primary_key == ("account_id", "provider", "provider_environment")
        and (
            "provider",
            "provider_environment",
            "external_customer_id",
        ) in unique_columns
    ):
        return
    environment_expression = (
        "COALESCE(NULLIF(TRIM(provider_environment), ''), 'Unknown')"
        if "provider_environment" in columns
        else "'Unknown'"
    )
    conn.execute(
        "ALTER TABLE billing_provider_customers "
        "RENAME TO billing_provider_customers_legacy_environment"
    )
    conn.execute(
        """
        CREATE TABLE billing_provider_customers (
            account_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_environment TEXT NOT NULL DEFAULT 'Unknown',
            external_customer_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(account_id, provider, provider_environment),
            UNIQUE(provider, provider_environment, external_customer_id),
            FOREIGN KEY(account_id) REFERENCES community_accounts(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        f"""
        INSERT INTO billing_provider_customers (
            account_id, provider, provider_environment, external_customer_id,
            created_at, updated_at
        )
        SELECT account_id, provider, {environment_expression}, external_customer_id,
               created_at, updated_at
        FROM billing_provider_customers_legacy_environment
        """
    )
    conn.execute("DROP TABLE billing_provider_customers_legacy_environment")


def _migrate_billing_subscription_environment_postgres(conn: Any) -> None:
    """Fail closed when upgrading rows created before environment isolation."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE billing_subscriptions "
            "ADD COLUMN IF NOT EXISTS provider_environment "
            "TEXT NOT NULL DEFAULT 'Unknown'"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS "
            "idx_billing_subscriptions_account_environment_status_end "
            "ON billing_subscriptions("
            "account_id, provider_environment, status, current_period_end)"
        )


def _migrate_search_quota_reservation_plan_sqlite(
    conn: sqlite3.Connection,
) -> None:
    """Snapshot the effective plan for historical provider-cost reporting."""
    _sqlite_add_missing_columns(
        conn,
        "search_quota_reservations",
        (("plan_code", "TEXT NOT NULL DEFAULT 'free'"),),
    )


def _migrate_search_quota_reservation_plan_postgres(conn: Any) -> None:
    """Snapshot the effective plan for historical provider-cost reporting."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE search_quota_reservations "
            "ADD COLUMN IF NOT EXISTS plan_code TEXT NOT NULL DEFAULT 'free'"
        )


def _migrate_search_quota_reservation_jobs_sqlite(conn: sqlite3.Connection) -> None:
    """Backfill the multi-job association table from the legacy single-job link."""
    conn.execute(
        """
        INSERT OR IGNORE INTO search_quota_reservation_jobs (
            reservation_id, generation_job_id, attached_at
        )
        SELECT id, generation_job_id, updated_at
        FROM search_quota_reservations
        WHERE generation_job_id IS NOT NULL
        """
    )


def _migrate_search_quota_reservation_jobs_postgres(conn: Any) -> None:
    """Backfill the multi-job association table from the legacy single-job link."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO search_quota_reservation_jobs (
                reservation_id, generation_job_id, attached_at
            )
            SELECT id, generation_job_id, updated_at
            FROM search_quota_reservations
            WHERE generation_job_id IS NOT NULL
            ON CONFLICT (reservation_id, generation_job_id) DO NOTHING
            """
        )


def _migrate_billing_provider_customer_environment_postgres(conn: Any) -> None:
    """Re-key legacy mappings so test and live Stripe IDs cannot collide."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE billing_provider_customers "
            "ADD COLUMN IF NOT EXISTS provider_environment "
            "TEXT NOT NULL DEFAULT 'Unknown'"
        )
        cur.execute(
            """
            SELECT conname, contype, pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conrelid = 'billing_provider_customers'::regclass
              AND contype IN ('p', 'u')
            """
        )
        constraints = list(cur.fetchall())
        normalized = [
            (str(row[0]), str(row[1]), str(row[2]).replace('"', "").lower())
            for row in constraints
        ]
        has_primary_key = any(
            kind == "p"
            and "(account_id, provider, provider_environment)" in definition
            for _name, kind, definition in normalized
        )
        has_scoped_customer_unique = any(
            kind == "u"
            and "(provider, provider_environment, external_customer_id)" in definition
            for _name, kind, definition in normalized
        )
        if has_primary_key and has_scoped_customer_unique:
            return
        for name, _kind, _definition in normalized:
            quoted_name = name.replace('"', '""')
            cur.execute(
                f'ALTER TABLE billing_provider_customers '
                f'DROP CONSTRAINT "{quoted_name}"'
            )
        cur.execute(
            "ALTER TABLE billing_provider_customers "
            "ADD CONSTRAINT billing_provider_customers_pkey "
            "PRIMARY KEY (account_id, provider, provider_environment)"
        )
        cur.execute(
            "ALTER TABLE billing_provider_customers "
            "ADD CONSTRAINT billing_provider_customers_provider_environment_customer_key "
            "UNIQUE (provider, provider_environment, external_customer_id)"
        )


def _migrate_durable_generation_foundation_sqlite(conn: sqlite3.Connection) -> None:
    """Idempotently upgrade legacy SQLite tables for durable generation work."""
    _sqlite_add_missing_columns(conn, "reel_generation_jobs", _DURABLE_JOB_SQLITE_COLUMNS)
    _sqlite_add_missing_columns(conn, "videos", _DURABLE_VIDEO_SQLITE_COLUMNS)
    _sqlite_add_missing_columns(conn, "reels", _DURABLE_REEL_SQLITE_COLUMNS)

    timestamp = now_iso()
    conn.execute(
        """
        UPDATE reel_generation_jobs
        SET status = 'cancelled',
            completed_at = COALESCE(completed_at, ?),
            updated_at = ?,
            terminal_error_code = 'legacy_job_contract',
            terminal_error_message = 'Legacy refinement work was replaced by durable generation jobs.'
        WHERE status IN ('queued', 'running')
          AND TRIM(COALESCE(content_fingerprint, '')) = ''
        """,
        (timestamp, timestamp),
    )
    conn.execute(
        """
        UPDATE reel_generation_jobs
        SET status = 'cancelled',
            completed_at = COALESCE(completed_at, ?),
            updated_at = CASE WHEN TRIM(COALESCE(updated_at, '')) = '' THEN ? ELSE updated_at END,
            terminal_error_code = COALESCE(terminal_error_code, 'superseded')
        WHERE status = 'superseded'
        """,
        (timestamp, timestamp),
    )
    active_rows = conn.execute(
        """
        SELECT id, request_key
        FROM reel_generation_jobs
        WHERE status IN ('queued', 'running')
        ORDER BY request_key,
                 CASE status WHEN 'running' THEN 0 ELSE 1 END,
                 created_at,
                 id
        """
    ).fetchall()
    seen_request_keys: set[str] = set()
    duplicate_ids: list[str] = []
    for job_id, request_key in active_rows:
        clean_key = str(request_key or "")
        if clean_key in seen_request_keys:
            duplicate_ids.append(str(job_id))
        else:
            seen_request_keys.add(clean_key)
    for job_id in duplicate_ids:
        conn.execute(
            """
            UPDATE reel_generation_jobs
            SET status = 'cancelled',
                completed_at = COALESCE(completed_at, ?),
                updated_at = ?,
                terminal_error_code = 'duplicate_active_request',
                terminal_error_message = 'Superseded while enforcing one active job per request key.'
            WHERE id = ? AND status IN ('queued', 'running')
            """,
            (timestamp, timestamp, job_id),
        )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_generation_jobs_active_request
        ON reel_generation_jobs(request_key)
        WHERE status IN ('queued', 'running')
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reel_generation_jobs_lease
        ON reel_generation_jobs(status, lease_expires_at, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reel_generation_jobs_assessment_plans
        ON reel_generation_jobs(
            learner_id, material_id, status,
            completed_at DESC, updated_at DESC, created_at DESC
        )
        """
    )


def _migrate_durable_generation_foundation_postgres(conn: Any) -> None:
    """Idempotently upgrade legacy PostgreSQL tables for durable generation work."""
    with conn.cursor() as cur:
        for name, declaration in _DURABLE_JOB_SQLITE_COLUMNS:
            cur.execute(
                f"ALTER TABLE reel_generation_jobs ADD COLUMN IF NOT EXISTS {name} {declaration}"
            )
        for name, declaration in _DURABLE_VIDEO_SQLITE_COLUMNS:
            cur.execute(f"ALTER TABLE videos ADD COLUMN IF NOT EXISTS {name} {declaration}")
        for name, declaration in _DURABLE_REEL_SQLITE_COLUMNS:
            cur.execute(f"ALTER TABLE reels ADD COLUMN IF NOT EXISTS {name} {declaration}")
        cur.execute("ALTER TABLE reel_generation_jobs ALTER COLUMN source_generation_id SET DEFAULT ''")
        cur.execute(
            """
            UPDATE reel_generation_jobs
            SET status = 'cancelled',
                completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP::text),
                updated_at = CURRENT_TIMESTAMP::text,
                terminal_error_code = 'legacy_job_contract',
                terminal_error_message = 'Legacy refinement work was replaced by durable generation jobs.'
            WHERE status IN ('queued', 'running')
              AND BTRIM(COALESCE(content_fingerprint, '')) = ''
            """
        )
        cur.execute(
            """
            UPDATE reel_generation_jobs
            SET status = 'cancelled',
                completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP::text),
                updated_at = CASE WHEN BTRIM(COALESCE(updated_at, '')) = '' THEN CURRENT_TIMESTAMP::text ELSE updated_at END,
                terminal_error_code = COALESCE(terminal_error_code, 'superseded')
            WHERE status = 'superseded'
            """
        )
        cur.execute(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY request_key
                           ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at, id
                       ) AS active_rank
                FROM reel_generation_jobs
                WHERE status IN ('queued', 'running')
            )
            UPDATE reel_generation_jobs
            SET status = 'cancelled',
                completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP::text),
                updated_at = CURRENT_TIMESTAMP::text,
                terminal_error_code = 'duplicate_active_request',
                terminal_error_message = 'Superseded while enforcing one active job per request key.'
            WHERE id IN (SELECT id FROM ranked WHERE active_rank > 1)
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_reel_generation_jobs_active_request
            ON reel_generation_jobs(request_key)
            WHERE status IN ('queued', 'running')
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reel_generation_jobs_lease
            ON reel_generation_jobs(status, lease_expires_at, created_at)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reel_generation_jobs_assessment_plans
            ON reel_generation_jobs(
                learner_id, material_id, status,
                completed_at DESC, updated_at DESC, created_at DESC
            )
            """
        )


def _ensure_reels_generation_index_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reels_generation_id ON reels(generation_id)")


def _ensure_reels_generation_index_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reels_generation_id ON reels(generation_id)")


def _connect_postgres_with_retry():
    """Connect to Postgres, retrying transient startup errors.

    Returns an open psycopg connection (the caller is responsible for
    closing / context-managing it). Raises the last exception if every
    retry attempt fails.
    """
    assert psycopg is not None
    last_exc: BaseException | None = None
    backoff = INIT_DB_INITIAL_BACKOFF_SEC
    for attempt in range(1, INIT_DB_MAX_RETRIES + 1):
        try:
            return psycopg.connect(_database_url(), connect_timeout=10)
        except Exception as exc:
            last_exc = exc
            if not _is_transient_postgres_startup_error(exc) or attempt >= INIT_DB_MAX_RETRIES:
                raise
            logger.warning(
                "init_db: Postgres not ready yet (attempt %d/%d): %s — retrying in %.1fs",
                attempt, INIT_DB_MAX_RETRIES, str(exc).splitlines()[0][:180], backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 1.6, INIT_DB_MAX_BACKOFF_SEC)
    # Should not reach here — loop either returns or raises — but keep mypy happy.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("init_db: exhausted retries without connecting")


def init_db() -> None:
    global _db_ready
    if _is_postgres_configured():
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set to PostgreSQL, but psycopg is not installed.")
        with _connect_postgres_with_retry() as conn:
            with conn.cursor() as cur:
                for statement in _postgres_schema_statements():
                    cur.execute(statement)
                # Lightweight schema migration for older community_sets tables.
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS email TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS email_normalized TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS verified_at TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS verification_code_hash TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS verification_expires_at TEXT")
                cur.execute("ALTER TABLE community_accounts ADD COLUMN IF NOT EXISTS legacy_claim_owner_key_hash TEXT")
                cur.execute("ALTER TABLE reels ADD COLUMN IF NOT EXISTS generation_id TEXT")
                cur.execute("ALTER TABLE reel_generation_jobs ADD COLUMN IF NOT EXISTS request_params_json TEXT NOT NULL DEFAULT '{}'")
                cur.execute("ALTER TABLE reel_generation_heads ADD COLUMN IF NOT EXISTS refinement_state_json TEXT NOT NULL DEFAULT '{}'")
                _ensure_reels_generation_index_postgres(conn)
                cur.execute(
                    "UPDATE community_accounts SET email = NULL WHERE email IS NOT NULL AND BTRIM(email) = ''"
                )
                cur.execute(
                    "UPDATE community_accounts SET email_normalized = NULL WHERE email_normalized IS NOT NULL AND BTRIM(email_normalized) = ''"
                )
                cur.execute(
                    """
                    UPDATE community_accounts
                    SET verified_at = created_at
                    WHERE (email IS NULL OR BTRIM(email) = '')
                      AND (verified_at IS NULL OR BTRIM(verified_at) = '')
                    """
                )
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_community_accounts_email_normalized_unique ON community_accounts(email_normalized)"
                )
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS updated_at TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS owner_key_hash TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS owner_account_id TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS visibility TEXT")
                cur.execute("ALTER TABLE community_sets ADD COLUMN IF NOT EXISTS dislikes INTEGER NOT NULL DEFAULT 0")
                cur.execute("ALTER TABLE community_material_history ADD COLUMN IF NOT EXISTS active_index INTEGER")
                cur.execute("ALTER TABLE community_material_history ADD COLUMN IF NOT EXISTS active_reel_id TEXT")
                cur.execute("ALTER TABLE community_material_history ADD COLUMN IF NOT EXISTS recall_json TEXT NOT NULL DEFAULT '{}'")
                cur.execute("ALTER TABLE community_account_settings ADD COLUMN IF NOT EXISTS autoplay_next_reel INTEGER NOT NULL DEFAULT 0")
                cur.execute("ALTER TABLE community_account_settings ADD COLUMN IF NOT EXISTS creative_commons_only INTEGER NOT NULL DEFAULT 0")
                cur.execute("UPDATE community_sets SET updated_at = created_at WHERE updated_at IS NULL OR BTRIM(updated_at) = ''")
                cur.execute(
                    """
                    UPDATE community_sets
                    SET owner_key_hash = %s
                    WHERE owner_key_hash IS NULL OR BTRIM(owner_key_hash) = ''
                    """,
                    (LEGACY_COMMUNITY_OWNER_HASH,),
                )
                cur.execute(
                    """
                    UPDATE community_sets
                    SET visibility = CASE
                        WHEN featured = 1 OR owner_key_hash = %s THEN %s
                        ELSE %s
                    END
                    WHERE visibility IS NULL OR BTRIM(visibility) = ''
                    """,
                    (LEGACY_COMMUNITY_OWNER_HASH, PUBLIC_COMMUNITY_VISIBILITY, DEFAULT_COMMUNITY_VISIBILITY),
                )
                cur.execute(
                    """
                    UPDATE community_sets
                    SET visibility = %s
                    WHERE owner_account_id IS NOT NULL
                      AND (
                          visibility IS NULL
                          OR BTRIM(visibility) = ''
                          OR visibility = 'private'
                      )
                    """,
                    (PUBLIC_COMMUNITY_VISIBILITY,),
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_community_sets_owner_account_updated_at ON community_sets(owner_account_id, updated_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_community_sets_visibility_featured_updated_at ON community_sets(visibility, featured DESC, updated_at DESC)"
                )
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS coverage_ratio REAL")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS cue_count INTEGER")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS source_kind TEXT")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS quality_score REAL")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS language TEXT")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS extractor_version TEXT")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS model_version TEXT")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS normalization_version TEXT")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS quality_status TEXT")
                cur.execute("ALTER TABLE transcript_cache ADD COLUMN IF NOT EXISTS quality_rejection_reason TEXT")
                cur.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS provider TEXT DEFAULT 'youtube'")
                cur.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS playback_url TEXT")
                cur.execute("ALTER TABLE materials ADD COLUMN IF NOT EXISTS knowledge_level TEXT NOT NULL DEFAULT 'beginner'")
                cur.execute("ALTER TABLE materials ADD COLUMN IF NOT EXISTS level_adjustment REAL NOT NULL DEFAULT 0.0")
                cur.execute("ALTER TABLE reels ADD COLUMN IF NOT EXISTS difficulty REAL")
                cur.execute("ALTER TABLE reels ADD COLUMN IF NOT EXISTS ai_summary TEXT NOT NULL DEFAULT ''")
                cur.execute("ALTER TABLE reels ADD COLUMN IF NOT EXISTS match_reason TEXT NOT NULL DEFAULT ''")
                cur.execute("ALTER TABLE reels ADD COLUMN IF NOT EXISTS informativeness REAL")
            _migrate_durable_generation_foundation_postgres(conn)
            _migrate_lesson_order_postgres(conn)
            _migrate_assessment_checkpoint_postgres(conn)
            _migrate_api_idempotency_postgres(conn)
            _migrate_billing_provider_customer_environment_postgres(conn)
            _migrate_billing_subscription_environment_postgres(conn)
            _migrate_search_quota_reservation_plan_postgres(conn)
            _migrate_search_quota_reservation_jobs_postgres(conn)
            _migrate_reels_unique_clip_index_postgres(conn)
            _migrate_reel_feedback_uniqueness_postgres(conn)
            _migrate_assessment_scroll_postgres(conn)
            conn.commit()
        _db_ready = True
        return

    conn = sqlite3.connect(_db_path())
    try:
        conn.executescript(SCHEMA)
        # Lightweight schema migration for existing local databases.
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN view_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN provider TEXT DEFAULT 'youtube'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE videos ADD COLUMN playback_url TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN email TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN email_normalized TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN verified_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN verification_code_hash TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN verification_expires_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_accounts ADD COLUMN legacy_claim_owner_key_hash TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE reels ADD COLUMN generation_id TEXT")
        except sqlite3.OperationalError:
            pass
        _ensure_reels_generation_index_sqlite(conn)
        try:
            conn.execute("ALTER TABLE reel_generation_jobs ADD COLUMN request_params_json TEXT NOT NULL DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE reel_generation_heads ADD COLUMN refinement_state_json TEXT NOT NULL DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("UPDATE community_accounts SET email = NULL WHERE email IS NOT NULL AND TRIM(email) = ''")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                "UPDATE community_accounts SET email_normalized = NULL WHERE email_normalized IS NOT NULL AND TRIM(email_normalized) = ''"
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_accounts
                SET verified_at = created_at
                WHERE (email IS NULL OR TRIM(email) = '')
                  AND (verified_at IS NULL OR TRIM(verified_at) = '')
                """
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN updated_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN owner_key_hash TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN owner_account_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN visibility TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_sets ADD COLUMN dislikes INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_material_history ADD COLUMN active_index INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_material_history ADD COLUMN active_reel_id TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_material_history ADD COLUMN recall_json TEXT NOT NULL DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_account_settings ADD COLUMN autoplay_next_reel INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE community_account_settings ADD COLUMN creative_commons_only INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("UPDATE community_sets SET updated_at = created_at WHERE updated_at IS NULL OR TRIM(updated_at) = ''")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_sets
                SET owner_key_hash = ?
                WHERE owner_key_hash IS NULL OR TRIM(owner_key_hash) = ''
                """,
                (LEGACY_COMMUNITY_OWNER_HASH,),
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_sets
                SET visibility = CASE
                    WHEN featured = 1 OR owner_key_hash = ? THEN ?
                    ELSE ?
                END
                WHERE visibility IS NULL OR TRIM(visibility) = ''
                """,
                (LEGACY_COMMUNITY_OWNER_HASH, PUBLIC_COMMUNITY_VISIBILITY, DEFAULT_COMMUNITY_VISIBILITY),
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute(
                """
                UPDATE community_sets
                SET visibility = ?
                WHERE owner_account_id IS NOT NULL
                  AND (
                      visibility IS NULL
                      OR TRIM(visibility) = ''
                      OR visibility = 'private'
                  )
                """,
                (PUBLIC_COMMUNITY_VISIBILITY,),
            )
        except sqlite3.OperationalError:
            pass
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_community_accounts_email_normalized_unique ON community_accounts(email_normalized)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_community_sets_owner_account_updated_at ON community_sets(owner_account_id, updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_community_sets_visibility_featured_updated_at ON community_sets(visibility, featured DESC, updated_at DESC)"
        )
        try:
            conn.execute("ALTER TABLE transcript_cache ADD COLUMN coverage_ratio REAL")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE transcript_cache ADD COLUMN cue_count INTEGER")
        except sqlite3.OperationalError:
            pass
        for _col_sql in [
            "ALTER TABLE transcript_cache ADD COLUMN source_kind TEXT",
            "ALTER TABLE transcript_cache ADD COLUMN quality_score REAL",
            "ALTER TABLE transcript_cache ADD COLUMN language TEXT",
            "ALTER TABLE transcript_cache ADD COLUMN extractor_version TEXT",
            "ALTER TABLE transcript_cache ADD COLUMN model_version TEXT",
            "ALTER TABLE transcript_cache ADD COLUMN normalization_version TEXT",
            "ALTER TABLE transcript_cache ADD COLUMN quality_status TEXT",
            "ALTER TABLE transcript_cache ADD COLUMN quality_rejection_reason TEXT",
        ]:
            try:
                conn.execute(_col_sql)
            except sqlite3.OperationalError:
                pass
        _migrate_durable_generation_foundation_sqlite(conn)
        _migrate_lesson_order_sqlite(conn)
        _migrate_assessment_checkpoint_sqlite(conn)
        _migrate_api_idempotency_sqlite(conn)
        _migrate_billing_provider_customer_environment_sqlite(conn)
        _migrate_billing_subscription_environment_sqlite(conn)
        _migrate_search_quota_reservation_plan_sqlite(conn)
        _migrate_search_quota_reservation_jobs_sqlite(conn)
        _migrate_reels_unique_clip_index_sqlite(conn)
        _migrate_knowledge_level_sqlite(conn)
        _migrate_reel_learning_content_sqlite(conn)
        _migrate_reel_feedback_uniqueness_sqlite(conn)
        _migrate_assessment_scroll_sqlite(conn)
        conn.commit()
    finally:
        conn.close()
    _db_ready = True


def ensure_db_initialized() -> None:
    global _db_ready
    if _db_ready:
        return
    with _db_init_lock:
        if _db_ready:
            return
        init_db()


## ---------------------------------------------------------------------------
## Lightweight PostgreSQL connection pool (no external dependency required)
## ---------------------------------------------------------------------------
_pg_pool_lock = threading.Lock()
_pg_pool: list[Any] = []
_PG_POOL_MAX_IDLE = 6


def _pg_pool_acquire() -> Any:
    """Return a reusable PostgreSQL connection, or create a fresh one."""
    while True:
        conn: Any = None
        with _pg_pool_lock:
            if _pg_pool:
                conn = _pg_pool.pop()
        if conn is None:
            return psycopg.connect(_database_url(), connect_timeout=15, autocommit=True)
        # Validate before reuse — discard stale/broken connections.
        try:
            conn.execute("SELECT 1")
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            # Loop and try next pooled connection or create a new one.


def _pg_pool_release(conn: Any) -> None:
    """Return a connection to the pool (or close it if the pool is full)."""
    try:
        # Reset to a clean state: rollback any uncommitted work, set autocommit.
        if not conn.autocommit:
            conn.rollback()
            conn.autocommit = True
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return
    with _pg_pool_lock:
        if len(_pg_pool) < _PG_POOL_MAX_IDLE:
            _pg_pool.append(conn)
        else:
            try:
                conn.close()
            except Exception:
                pass


@contextmanager
def get_conn(*, transactional: bool = False):
    ensure_db_initialized()
    if _is_postgres_configured():
        if psycopg is None:
            raise RuntimeError("DATABASE_URL is set to PostgreSQL, but psycopg is not installed.")
        conn = _pg_pool_acquire()
        conn.autocommit = not transactional
        try:
            yield conn
        except Exception:
            if transactional:
                conn.rollback()
            raise
        else:
            if transactional:
                conn.commit()
        finally:
            _pg_pool_release(conn)
        return

    busy_timeout_ms = _sqlite_busy_timeout_ms()
    conn = sqlite3.connect(
        _db_path(),
        timeout=max(1.0, busy_timeout_ms / 1000.0),
        isolation_level="" if transactional else None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    except Exception:
        if transactional:
            conn.rollback()
        raise
    else:
        if transactional:
            conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert(conn: Any, table: str, data: dict[str, Any]) -> None:
    """Plain INSERT — raises DatabaseIntegrityError on unique constraint violation."""
    if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(table):
        raise ValueError(f"Unsafe table name: {table!r}")
    cols = list(data.keys())
    for col in cols:
        if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(col):
            raise ValueError(f"Unsafe column name: {col!r}")
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    values = [data[c] for c in cols]
    is_pg = _is_postgres_conn(conn)
    query = _adapt_query_for_postgres(sql) if is_pg else sql
    try:
        if is_pg:
            with conn.cursor() as cur:
                if getattr(conn, "autocommit", False):
                    cur.execute(query, values)
                else:
                    cur.execute("SAVEPOINT studyreels_insert")
                    try:
                        cur.execute(query, values)
                    except Exception:
                        cur.execute("ROLLBACK TO SAVEPOINT studyreels_insert")
                        cur.execute("RELEASE SAVEPOINT studyreels_insert")
                        raise
                    else:
                        cur.execute("RELEASE SAVEPOINT studyreels_insert")
        else:
            conn.execute(query, values)
    except Exception as exc:
        if is_pg and _is_unique_violation(exc):
            raise DatabaseIntegrityError(str(exc)) from exc
        if isinstance(exc, sqlite3.IntegrityError) and _is_sqlite_unique_violation(exc):
            raise DatabaseIntegrityError(str(exc)) from exc
        raise


def _is_unique_violation(exc: Exception) -> bool:
    if pg_errors is not None and isinstance(exc, pg_errors.UniqueViolation):
        return True
    return getattr(exc, "sqlstate", "") == "23505"


def _is_sqlite_unique_violation(exc: sqlite3.IntegrityError) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    unique_codes = {
        getattr(sqlite3, "SQLITE_CONSTRAINT_PRIMARYKEY", -1),
        getattr(sqlite3, "SQLITE_CONSTRAINT_UNIQUE", -1),
    }
    if error_code in unique_codes:
        return True
    message = str(exc).lower()
    return "unique constraint failed" in message or "is not unique" in message


def upsert(conn: Any, table: str, data: dict[str, Any], pk: str | list[str] = "id") -> None:
    if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(table):
        raise ValueError(f"Unsafe table name: {table!r}")
    # Normalize pk to a list for composite key support.
    pk_cols: list[str] = [pk] if isinstance(pk, str) else list(pk)
    for p in pk_cols:
        if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(p):
            raise ValueError(f"Unsafe primary key name: {p!r}")
    pk_set = set(pk_cols)
    cols = list(data.keys())
    for col in cols:
        if not _SAFE_SQL_IDENTIFIER_RE.fullmatch(col):
            raise ValueError(f"Unsafe column name: {col!r}")
    placeholders = ", ".join(["?"] * len(cols))
    assignments = ", ".join([f"{c}=excluded.{c}" for c in cols if c not in pk_set])
    conflict_cols = ", ".join(pk_cols)
    if assignments:
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict_cols}) DO UPDATE SET {assignments}"
        )
    else:
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT({conflict_cols}) DO NOTHING"

    values = [data[c] for c in cols]
    is_pg = _is_postgres_conn(conn)
    query = _adapt_query_for_postgres(sql) if is_pg else sql

    for attempt in range(8):
        try:
            if is_pg:
                with conn.cursor() as cur:
                    if getattr(conn, "autocommit", False):
                        cur.execute(query, values)
                    else:
                        cur.execute("SAVEPOINT studyreels_upsert")
                        try:
                            cur.execute(query, values)
                        except Exception:
                            cur.execute("ROLLBACK TO SAVEPOINT studyreels_upsert")
                            cur.execute("RELEASE SAVEPOINT studyreels_upsert")
                            raise
                        else:
                            cur.execute("RELEASE SAVEPOINT studyreels_upsert")
            else:
                conn.execute(query, values)
            return
        except sqlite3.OperationalError as exc:
            # WAL can transiently lock under concurrent writes; retry quickly.
            if "locked" not in str(exc).lower() or attempt >= 7:
                raise
            try:
                conn.rollback()
            except Exception:
                pass
            # Exponential-ish backoff helps during concurrent material ingestion.
            time.sleep(min(2.0, 0.05 * (2 ** attempt)))
        except Exception as exc:
            if is_pg and _is_unique_violation(exc):
                raise DatabaseIntegrityError(str(exc)) from exc
            raise


def fetch_all(conn: Any, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    if _is_postgres_conn(conn):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_adapt_query_for_postgres(query), tuple(params))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def fetch_one(conn: Any, query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    if _is_postgres_conn(conn):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_adapt_query_for_postgres(query), tuple(params))
            row = cur.fetchone()
        return dict(row) if row else None

    row = conn.execute(query, tuple(params)).fetchone()
    return dict(row) if row else None


def execute_modify(conn: Any, query: str, params: Iterable[Any] = ()) -> int:
    """Execute a non-SELECT statement (INSERT/UPDATE/DELETE) and return the number of affected rows."""
    if _is_postgres_conn(conn):
        with conn.cursor() as cur:
            cur.execute(_adapt_query_for_postgres(query), tuple(params))
            return int(cur.rowcount or 0)

    cursor = conn.execute(query, tuple(params))
    return int(getattr(cursor, "rowcount", 0) or 0)


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
