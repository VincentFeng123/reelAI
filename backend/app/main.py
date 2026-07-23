import asyncio
import hashlib
import json
import ipaddress
import logging
import math
import os
import re
import secrets
import smtplib
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parseaddr
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Literal
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute

from backend import config as pipeline_config
from backend.concept_families import (
    has_incompatible_gemini_concept_family_contract,
)
from backend.concept_tokens import semantic_key, semantic_tokens
from backend.intent_obligations import intent_obligation_keys

from .config import get_settings
from .db import (
    DEFAULT_COMMUNITY_VISIBILITY,
    DatabaseIntegrityError,
    LEGACY_COMMUNITY_OWNER_HASH,
    LEGACY_LEARNER_ID,
    PUBLIC_COMMUNITY_VISIBILITY,
    dumps_json,
    execute_modify,
    fetch_all,
    fetch_one,
    get_conn,
    init_db,
    insert,
    is_transient_postgres_transaction_error,
    now_iso,
    upsert,
)
from .models import (
    AssessmentAnswerRequest,
    AssessmentAnswerResponse,
    AssessmentNextRequest,
    AssessmentSnoozeResponse,
    AssessmentWrapperResponse,
    BillingCheckoutRequest,
    BillingPlansResponse,
    BillingRedirectResponse,
    BillingStatusResponse,
    ChatRequest,
    ChatResponse,
    CommunityAuthMeResponse,
    CommunityAuthLoginRequest,
    CommunityAuthRegisterRequest,
    CommunityChangeEmailRequest,
    CommunityChangeEmailResponse,
    CommunityDeleteAccountRequest,
    CommunityAuthSessionResponse,
    CommunityAccountOut,
    CommunityChangePasswordRequest,
    CommunityDraftPayload,
    CommunityDraftsResponse,
    CommunityFeedSnapshotPayload,
    CommunityFeedSnapshotsResponse,
    CommunityMaterialGroupsPayload,
    CommunityMaterialGroupsResponse,
    CommunityMaterialSeedsPayload,
    CommunityMaterialSeedsResponse,
    CommunityResendVerificationResponse,
    CommunitySendSignupVerificationRequest,
    CommunitySendSignupVerificationResponse,
    CommunityHistoryItemOut,
    CommunityHistoryReplaceRequest,
    CommunityHistoryResponse,
    CommunitySettingsPayload,
    CommunitySettingsResponse,
    CommunityStarredSetsPayload,
    CommunityStarredSetsResponse,
    CommunityReelOut,
    CommunitySetCreateRequest,
    CommunitySetFeedbackRequest,
    CommunitySetFeedbackResponse,
    CommunitySetOut,
    CommunitySetUpdateRequest,
    CommunitySetsDeleteRequest,
    CommunitySetsResponse,
    CommunityVerifyAccountRequest,
    CommunityVerifyAccountResponse,
    CommunityVerifySignupEmailRequest,
    CommunityVerifySignupEmailResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedResponse,
    GenerationJobQueuedResponse,
    GenerationJobStatusResponse,
    MaterialLevelUpdateRequest,
    MaterialResponse,
    ReelsCanGenerateAnyRequest,
    ReelsCanGenerateAnyResponse,
    ReelsCanGenerateResponse,
    ReelsGenerateRequest,
    ReelsGenerateResponse,
    ReelProgressRequest,
    ReelProgressResponse,
    ReelScrollResponse,
)
from .services import llm_router
from .services.assessments import (
    ACTIVE_REEL_OPEN_FRACTION,
    AssessmentCancelledError,
    AssessmentService,
    assessment_checkpoint_reel_ids,
)
from .services.email import send_welcome_email
from .services.embeddings import EmbeddingService
from .services.material_intelligence import MaterialIntelligenceService
from .services.billing import (
    DailySearchLimitReached,
    attach_reservation_to_job,
    billing_enforcement_enabled,
    billing_status,
    plans_payload,
    reserve_search,
    settle_operation,
)
from .services.billing_providers import (
    BillingAccountNotFoundError,
    BillingConfigurationError,
    BillingVerificationError,
    DuplicateSubscriptionError,
    cancel_stripe_for_account,
    construct_stripe_event,
    create_stripe_checkout,
    create_stripe_portal,
    lock_billing_account,
    process_stripe_event,
)
from .services.lesson_ordering import (
    LESSON_ORDER_MAX_CLIPS,
    LESSON_ORDER_PROMPT_VERSION,
    _filter_same_source_overlaps,
    order_lesson_batch,
)
from .services.knowledge_level import effective_level_target
from .services.idempotency import (
    IdempotencyConflictError,
    complete_idempotency_key,
    lock_idempotency_attempt,
    normalize_idempotency_key,
    release_idempotency_key,
    request_fingerprint as build_idempotency_fingerprint,
    reserve_idempotency_key,
)
from .services.parsers import ParseError, extract_text_from_file
from .services.reels import GenerationCancelledError, ReelService
from .services.search_query_plan import build_search_query_plan
from .services.storage import get_storage
from .services.text_utils import chunk_text, normalize_whitespace

from .ingestion.errors import (
    DownloadError as IngestDownloadError,
    IngestError,
    RateLimitedError as IngestRateLimitedError,
    SegmentationError as IngestSegmentationError,
    ServerlessUnavailable as IngestServerlessUnavailable,
    TranscriptionError as IngestTranscriptionError,
    UnsupportedSourceError as IngestUnsupportedSourceError,
)
from .clip_engine.errors import (
    CancellationError as ClipEngineCancellationError,
    EngineError as ClipEngineError,
    JOB_GLOBAL_PROVIDER_ERROR_CODES,
    ProviderRateLimitError,
)
from .clip_engine.errors import ProviderError as ClipEngineProviderError
from .clip_engine.provider_cache import DatabaseProviderCache, TRANSCRIPT_SCHEMA_VERSION
from .clip_engine.provider_cache import normalize_filters as normalize_provider_filters
from .clip_engine.provider_cache import search_cache_key
from .clip_engine.provider_runtime import (
    GenerationContext,
    ProviderUsageRecord,
    gemini_usage_records_exposure,
)
from .clip_engine.clipper.supadata_client import fetch_transcript_artifact
from .clip_engine.supadata_search import search_one as supadata_search_one
from .clip_engine.metadata import canonicalize_youtube_url, normalize_youtube_video_id
from .clip_engine.silence import persisted_boundary_is_usable
from .ingestion.models import (
    IngestFeedRequest,
    IngestFeedResult,
    IngestRequest,
    IngestResult,
    IngestSearchRequest,
    IngestSearchResult,
    IngestTopicCutRequest,
    IngestTopicCutResult,
)
from .ingestion.pipeline import IngestionPipeline
from .services.generation_jobs import (
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_LEASE_SECONDS,
    DURABLE_QUEUE_WAIT_PARAM as GENERATION_DURABLE_QUEUE_WAIT_PARAM,
    EMPTY_ADAPTATION_FINGERPRINT as GENERATION_EMPTY_ADAPTATION_FINGERPRINT,
    REQUEST_SCHEMA_VERSION as GENERATION_REQUEST_SCHEMA_VERSION,
    JobLeaseLostError,
    TERMINAL_STATUSES as GENERATION_TERMINAL_STATUSES,
    admit_gemini_dispatch_ticket,
    append_event as append_generation_event,
    build_request_key as build_durable_request_key,
    cancellation_requested as generation_cancellation_requested,
    checkpoint_yielded_attempt_usage as checkpoint_generation_yielded_usage,
    expire_stale_queued_job as expire_stale_generation_job,
    find_active_job as find_active_generation_job,
    find_completed_job as find_completed_generation_job,
    get_job as get_generation_job,
    heartbeat_job as heartbeat_generation_job,
    lease_next_job,
    material_content_fingerprint,
    next_queued_retry_delay,
    record_provider_usage,
    requeue_retryable_failure as requeue_generation_retryable_failure,
    replay_events as replay_generation_events,
    request_cancellation as request_generation_cancellation,
    settle_gemini_dispatch_ticket,
    submit_or_get_active as submit_generation_job,
    transition_terminal as transition_generation_terminal,
    update_progress as update_generation_progress,
)

settings = get_settings()
logger = logging.getLogger(__name__)

GENERATION_LEASE_SEC = max(
    2,
    min(DEFAULT_LEASE_SECONDS, int(settings.generation_job_lease_sec)),
)
GENERATION_HEARTBEAT_SEC = max(
    1,
    min(
        int(settings.generation_job_heartbeat_sec or DEFAULT_HEARTBEAT_SECONDS),
        GENERATION_LEASE_SEC // 2,
    ),
)
GENERATION_WORKER_POLL_SEC = max(
    900.0,
    min(3600.0, float(settings.generation_job_poll_sec)),
)
# One job already analyzes its Fast/Slow source budget concurrently. Keep one
# process worker so a stale Railway environment override cannot multiply CPU,
# RAM, or Gemini spend on the Hobby deployment.
GENERATION_WORKER_COUNT = 1
GENERATION_DB_TRANSACTION_MAX_ATTEMPTS = 2
_generation_worker_ids = tuple(
    f"worker-{uuid.uuid4()}" for _index in range(GENERATION_WORKER_COUNT)
)
_generation_worker_id = _generation_worker_ids[0]
_generation_worker_stop = threading.Event()
_generation_worker_wake = threading.Event()
_generation_worker_lock = threading.Lock()
_generation_worker_thread: threading.Thread | None = None
_generation_worker_threads: list[threading.Thread] = []

@asynccontextmanager
async def lifespan(app_instance):
    os.makedirs(settings.data_dir, exist_ok=True)
    init_db()
    _start_generation_worker()
    _warn_if_hosted_auth_email_is_unconfigured()
    # The punctuation-restoration pipeline is NOT warmed here anymore: the
    # gemini clip engine never punctuates, and the only consumer chain
    # (_create_reel -> _trim_structural_edges_from_clip) has no callers since
    # the engine swap. Eager-loading its ~2.5GB model tripled idle RSS for
    # nothing; ReelService._get_punct_pipeline still lazy-loads if ever used.
    try:
        yield
    finally:
        _stop_generation_worker()

app = FastAPI(title="StudyReels API", version="0.1.0", lifespan=lifespan)


def _normalize_origin_candidate(raw_origin: str) -> str | None:
    clean = str(raw_origin or "").strip().rstrip("/")
    if not clean:
        return None
    if "://" not in clean:
        clean = f"https://{clean}"
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def _iter_configured_origin_candidates() -> list[str]:
    candidates: list[str] = []
    raw_frontend_origins = os.getenv("FRONTEND_ORIGINS", "")
    if raw_frontend_origins:
        candidates.extend(origin.strip() for origin in raw_frontend_origins.split(","))
    for env_name in (
        "NEXT_PUBLIC_SITE_URL",
        "NEXT_PUBLIC_APP_URL",
        "NEXT_PUBLIC_WEB_URL",
        "PUBLIC_URL",
        "APP_URL",
        "SITE_URL",
        "WEB_URL",
        "FRONTEND_URL",
        "VERCEL_URL",
        "VERCEL_BRANCH_URL",
        "VERCEL_PROJECT_PRODUCTION_URL",
        "RAILWAY_PUBLIC_DOMAIN",
        "RAILWAY_STATIC_URL",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            candidates.append(value)
    return candidates


def _is_hosted_runtime() -> bool:
    app_env = settings.app_env.strip().lower()
    return bool(
        os.getenv("VERCEL")
        or os.getenv("VERCEL_URL")
        or os.getenv("RAILWAY_ENVIRONMENT")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or os.getenv("RAILWAY_STATIC_URL")
        or os.getenv("K_SERVICE")
        or app_env in {"prod", "production", "staging"}
    )


def _build_allowed_origins() -> list[str]:
    local_defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
    candidates = [settings.frontend_origin, *local_defaults, *_iter_configured_origin_candidates()]
    normalized: list[str] = []
    seen: set[str] = set()
    for origin in candidates:
        clean = _normalize_origin_candidate(origin)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


allowed_origins = _build_allowed_origins()
allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip() or None
if allow_origin_regex is None and _is_hosted_runtime():
    # Build a project-scoped CORS regex from configured domains instead of
    # allowing every *.vercel.app / *.railway.app subdomain, which would let
    # any attacker with a deployment on those platforms make credentialed
    # cross-origin requests.
    _cors_project_slugs: list[str] = []
    for _cors_env in ("VERCEL_PROJECT_PRODUCTION_URL", "VERCEL_URL", "VERCEL_BRANCH_URL"):
        _cors_val = os.getenv(_cors_env, "").strip().lower()
        if _cors_val:
            # e.g. "my-project.vercel.app" → match "*.my-project.vercel.app"
            _cors_project_slugs.append(re.escape(_cors_val))
    for _cors_env in ("RAILWAY_PUBLIC_DOMAIN", "RAILWAY_STATIC_URL"):
        _cors_val = os.getenv(_cors_env, "").strip().lower()
        if _cors_val:
            _cors_project_slugs.append(re.escape(_cors_val))
    if _cors_project_slugs:
        _cors_alts = "|".join(_cors_project_slugs)
        allow_origin_regex = rf"^https://(?:[A-Za-z0-9-]+\.)*(?:{_cors_alts})$"
    # If no project domains are detected, fall back to explicit origins only
    # (already collected in allowed_origins from the same env vars).

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
# The Python service runs as a durable Railway process. Next.js remains on
# Vercel and proxies API traffic here; request-scoped serverless execution is
# intentionally unsupported for provider-backed generation.
SERVERLESS_MODE = False
ingestion_pipeline = IngestionPipeline(
    embedding_service=embedding_service,
    settings=settings,
    serverless_mode=SERVERLESS_MODE,
)
reel_service = ReelService(
    embedding_service=embedding_service,
    ingestion_pipeline=ingestion_pipeline,
)
assessment_service = AssessmentService()


# Each worker job remains tightly bounded; continuation jobs create additional
# unseen batches instead of imposing a terminal material-wide inventory cap.
REEL_BATCH_SIZE = 3
INITIAL_READY_BATCH_COUNT = 3
INITIAL_READY_REEL_TARGET = REEL_BATCH_SIZE * INITIAL_READY_BATCH_COUNT
GENERATION_OUTPUT_CEILINGS = {
    "fast": INITIAL_READY_REEL_TARGET,
    "slow": INITIAL_READY_REEL_TARGET,
}
GENERATION_SOURCE_BUDGETS = {"fast": 2, "slow": 3}
# The selector schema accepts at most 40 clips from one analyzed source. The
# organizer sees every persisted candidate from the unchanged source pass plus
# at most eight unseen source-chain reels that triggered a top-up. This remains
# under its own 200-ID response schema without widening acquisition or analysis.
GEMINI_SELECTOR_MAX_CLIPS_PER_SOURCE = 40
LESSON_ORDER_CANDIDATE_LIMITS = {
    mode: min(
        LESSON_ORDER_MAX_CLIPS,
        GEMINI_SELECTOR_MAX_CLIPS_PER_SOURCE * source_budget
        + INITIAL_READY_REEL_TARGET
        - 1,
    )
    for mode, source_budget in GENERATION_SOURCE_BUDGETS.items()
}
# Keep raw learner-signal evidence bounded before the organizer applies its
# existing tighter prompt limits. This lane lets Gemini compare semantically
# equivalent concepts even when separate valid calls use different family text.
LESSON_SIGNAL_HISTORY_LIMIT = 80
SOURCE_ANALYSIS_MAX_ATTEMPTS = 2
SELECTION_CONTRACT_VERSION = "quality_silence_v41"

VALID_VIDEO_DURATION_PREFS = {"any", "short", "medium", "long"}
VALID_SEARCH_INPUT_MODES = {"topic", "source", "file"}
DEFAULT_TARGET_CLIP_DURATION_SEC = 55
MIN_TARGET_CLIP_DURATION_SEC = 15
MAX_TARGET_CLIP_DURATION_SEC = 180
MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC = 15
DEFAULT_SETTINGS_MIN_RELEVANCE_THRESHOLD = 0.3
MIN_SETTINGS_MIN_RELEVANCE_THRESHOLD = 0.0
MAX_SETTINGS_MIN_RELEVANCE_THRESHOLD = 0.6
DEFAULT_SETTINGS_DEFAULT_INPUT_MODE = "source"
DEFAULT_SETTINGS_START_MUTED = True
DEFAULT_SETTINGS_PREFERRED_VIDEO_DURATION = "any"
DEFAULT_SETTINGS_TARGET_CLIP_DURATION_MIN_SEC = 20
DEFAULT_SETTINGS_TARGET_CLIP_DURATION_MAX_SEC = 55
MAX_COMMUNITY_REEL_DURATION_SEC = 8 * 60 * 60
COMMUNITY_REEL_DURATION_TIMEOUT_SEC = 6.0
MAX_DURATION_FETCH_REDIRECTS = 4
PRIVATE_HOST_SUFFIXES = (".localhost", ".local", ".internal")
COMMUNITY_OWNER_HEADER = "x-studyreels-owner-key"
COMMUNITY_OWNER_KEY_MIN_LENGTH = 24
COMMUNITY_SESSION_HEADER = "x-studyreels-session-token"
COMMUNITY_SESSION_TOKEN_MIN_LENGTH = 24
COMMUNITY_ACCOUNT_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,30}[A-Za-z0-9]$")
COMMUNITY_ACCOUNT_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COMMUNITY_ACCOUNT_PASSWORD_ITERATIONS = 240_000
COMMUNITY_SESSION_TTL_DAYS = 45
COMMUNITY_MAX_SESSIONS_PER_ACCOUNT = 20
COMMUNITY_VERIFICATION_CODE_LENGTH = 6
COMMUNITY_VERIFICATION_TTL_MINUTES = 20
COMMUNITY_REGISTER_CONFLICT_DETAIL = "An account with that username or email already exists."
COMMUNITY_CHANGE_EMAIL_CONFLICT_DETAIL = "Could not update the verification email."
MAX_COMMUNITY_THUMBNAIL_DATA_URL_BODY_CHARS = 2_000_000
RATE_LIMIT_WINDOW_SEC = 60.0
CHAT_RATE_LIMIT_PER_WINDOW = 20
MATERIAL_RATE_LIMIT_PER_WINDOW = 8
REELS_RATE_LIMIT_PER_WINDOW = 12
REELS_GENERATE_RATE_LIMIT_PER_WINDOW = 6
FEED_RATE_LIMIT_PER_WINDOW = 36
GENERATION_JOB_STATUS_RATE_LIMIT_PER_WINDOW = 120
FEEDBACK_RATE_LIMIT_PER_WINDOW = 60
ASSESSMENT_PROGRESS_RATE_LIMIT_PER_WINDOW = 240
ASSESSMENT_ACTION_RATE_LIMIT_PER_WINDOW = 60
COMMUNITY_WRITE_RATE_LIMIT_PER_WINDOW = 12
COMMUNITY_DURATION_RATE_LIMIT_PER_WINDOW = 30
COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW = 20
COMMUNITY_LOGIN_PER_USERNAME_RATE_LIMIT = 8
COMMUNITY_VERIFY_PER_ACCOUNT_RATE_LIMIT = 5
COMMUNITY_HISTORY_RATE_LIMIT_PER_WINDOW = 90
COMMUNITY_SETTINGS_RATE_LIMIT_PER_WINDOW = 90
INGEST_URL_RATE_LIMIT_PER_WINDOW = 6
INGEST_FEED_RATE_LIMIT_PER_WINDOW = 2
INGEST_SEARCH_RATE_LIMIT_PER_WINDOW = 3
# Topic-cut runs timestamped-transcript retrieval plus Gemini segmentation for one
# video. Six requests per window matches /api/ingest/url's full-video budget.
INGEST_TOPIC_CUT_RATE_LIMIT_PER_WINDOW = 6
MAX_COMMUNITY_HISTORY_ITEMS = 120
_rate_limit_lock = threading.Lock()
_rate_limit_hits: dict[str, deque[float]] = {}
_rate_limit_last_sweep = 0.0
_rate_limit_last_db_cleanup = 0.0


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def _is_public_host(host: str) -> bool:
    safe_host = host.strip().strip(".").lower()
    if not safe_host:
        return False
    if safe_host == "localhost" or safe_host.endswith(PRIVATE_HOST_SUFFIXES):
        return False
    try:
        ip = ipaddress.ip_address(safe_host)
    except ValueError:
        return True
    return bool(getattr(ip, "is_global", False))


def _normalize_public_http_url(raw_url: str, field_name: str) -> str:
    normalized = str(raw_url or "").strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an absolute http(s) URL.")
    host = (parsed.hostname or "").lower()
    if not _is_public_host(host):
        raise HTTPException(status_code=400, detail=f"{field_name} host is not allowed.")
    return urlunparse(parsed._replace(fragment=""))


def _normalize_community_thumbnail_url(raw_url: str) -> str:
    normalized = str(raw_url or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Thumbnail is required.")
    if normalized.startswith("/"):
        parsed = urlparse(normalized)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
            raise HTTPException(status_code=400, detail="thumbnail_url must be a same-origin path, safe data image, or public http(s) URL.")
        return urlunparse(("", "", parsed.path, "", parsed.query, ""))
    if normalized.lower().startswith("data:"):
        header, separator, body = normalized.partition(",")
        header_lower = header.lower()
        media_type = header_lower[5:].split(";", 1)[0]
        if separator != "," or media_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"} or ";base64" not in header_lower:
            raise HTTPException(status_code=400, detail="thumbnail_url must be a safe image data URL.")
        if not body.strip():
            raise HTTPException(status_code=400, detail="thumbnail_url image data is empty.")
        if len(body) > MAX_COMMUNITY_THUMBNAIL_DATA_URL_BODY_CHARS:
            raise HTTPException(status_code=400, detail="thumbnail_url image data is too large.")
        return normalized
    return _normalize_public_http_url(normalized, "thumbnail_url")


def _parse_ip_literal(raw_value: str) -> str | None:
    candidate = str(raw_value or "").strip()
    if not candidate:
        return None
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


# Trusted reverse-proxy CIDRs whose X-Forwarded-For / CF-Connecting-IP / X-Real-IP
# headers we will honor. Anything not in this list is treated as a direct peer and
# any forwarded-IP headers are ignored — this prevents attackers from spoofing
# `X-Forwarded-For: 1.2.3.4` to escape per-IP rate limits when the service is
# exposed directly. Configure via the `TRUSTED_PROXY_CIDRS` env var as a
# comma-separated list. Loopback and RFC1918 ranges are always trusted so local
# development and intra-cluster traffic continues to work.
_LOCAL_TRUSTED_CIDRS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _load_trusted_proxy_cidrs() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    raw = str(os.getenv("TRUSTED_PROXY_CIDRS") or "").strip()
    extra: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    if raw:
        for token in raw.split(","):
            clean = token.strip()
            if not clean:
                continue
            try:
                extra.append(ipaddress.ip_network(clean, strict=False))
            except ValueError:
                logger.warning("Ignoring invalid TRUSTED_PROXY_CIDRS entry: %s", clean)
    return (*_LOCAL_TRUSTED_CIDRS, *extra)


_TRUSTED_PROXY_CIDRS = _load_trusted_proxy_cidrs()


def _peer_is_trusted_proxy(peer_host: str) -> bool:
    candidate = str(peer_host or "").strip()
    if not candidate:
        return False
    try:
        peer_ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return any(peer_ip in net for net in _TRUSTED_PROXY_CIDRS)


def _extract_forwarded_ip(request: Request) -> str | None:
    for header_name in ("cf-connecting-ip", "x-real-ip"):
        raw = request.headers.get(header_name, "").strip()
        if not raw:
            continue
        parsed_ip = _parse_ip_literal(raw)
        if parsed_ip and _is_public_host(parsed_ip):
            return parsed_ip

    raw_forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if not raw_forwarded_for:
        return None
    candidates = [candidate.strip() for candidate in raw_forwarded_for.split(",")]
    # X-Forwarded-For is ordered client first, then each proxy hop that appended
    # itself later in the chain.
    for candidate in candidates:
        parsed_ip = _parse_ip_literal(candidate)
        if parsed_ip and _is_public_host(parsed_ip):
            return parsed_ip
    for candidate in candidates:
        parsed_ip = _parse_ip_literal(candidate)
        if parsed_ip:
            return parsed_ip
    return None


def _client_ip(request: Request) -> str:
    peer_host = request.client.host if request.client and request.client.host else ""
    if peer_host and _peer_is_trusted_proxy(peer_host):
        # Only honor forwarded-IP headers from trusted proxies. Attackers with a
        # direct connection to the server cannot forge X-Forwarded-For to escape
        # rate limits.
        forwarded = _extract_forwarded_ip(request)
        if forwarded:
            return forwarded
        return peer_host
    if peer_host:
        return peer_host
    return "unknown"


def _sweep_rate_limit_buckets(now: float, window_sec: float) -> None:
    global _rate_limit_last_sweep
    if now - _rate_limit_last_sweep < max(window_sec, 30.0):
        return
    cutoff = now - window_sec
    stale_keys: list[str] = []
    for current_key, current_bucket in _rate_limit_hits.items():
        while current_bucket and current_bucket[0] <= cutoff:
            current_bucket.popleft()
        if not current_bucket:
            stale_keys.append(current_key)
    for stale_key in stale_keys:
        _rate_limit_hits.pop(stale_key, None)
    _rate_limit_last_sweep = now


def _check_rate_limit_key(key: str, *, limit: int, window_sec: float) -> None:
    """Core rate limit check against a pre-built key. Raises HTTPException on breach."""
    if limit <= 0:
        return
    if _uses_persistent_rate_limits():
        _check_rate_limit_key_persistent(key, limit=limit, window_sec=window_sec)
        return
    now = time.monotonic()
    with _rate_limit_lock:
        _sweep_rate_limit_buckets(now, window_sec)
        bucket = _rate_limit_hits.get(key)
        if bucket is None:
            bucket = deque()
            _rate_limit_hits[key] = bucket
        cutoff = now - window_sec
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if not bucket:
            _rate_limit_hits.pop(key, None)
            bucket = deque()
            _rate_limit_hits[key] = bucket
        if len(bucket) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests. Please wait and try again.")
        bucket.append(now)


def _uses_persistent_rate_limits() -> bool:
    database_url = (settings.database_url or "").strip() or os.getenv("DATABASE_URL", "").strip()
    return database_url.startswith(("postgres://", "postgresql://"))


def _check_rate_limit_key_persistent(key: str, *, limit: int, window_sec: float) -> None:
    global _rate_limit_last_db_cleanup

    now_wall = time.time()
    cutoff = now_wall - window_sec
    should_cleanup = False
    with _rate_limit_lock:
        if now_wall - _rate_limit_last_db_cleanup >= max(window_sec, 30.0):
            _rate_limit_last_db_cleanup = now_wall
            should_cleanup = True

    with get_conn(transactional=True) as conn:
        # PostgreSQL advisory locks serialize checks per key across instances.
        fetch_one(conn, "SELECT pg_advisory_xact_lock(hashtext(?)) AS locked", (key,))
        if should_cleanup:
            execute_modify(conn, "DELETE FROM rate_limit_events WHERE hit_at <= ?", (cutoff,))
        row = fetch_one(
            conn,
            """
            SELECT COUNT(*) AS hit_count
            FROM rate_limit_events
            WHERE rate_key = ?
              AND hit_at > ?
            """,
            (key, cutoff),
        )
        hit_count = _to_int(row.get("hit_count") if row else 0, 0)
        if hit_count >= limit:
            raise HTTPException(status_code=429, detail="Too many requests. Please wait and try again.")
        insert(
            conn,
            "rate_limit_events",
            {
                "id": str(uuid.uuid4()),
                "rate_key": key,
                "hit_at": now_wall,
                "created_at": now_iso(),
            },
        )


def _rate_limit_identity_key(request: Request) -> str:
    """
    Preferred rate-limit bucket key for a request. Uses the owner key hash when
    present (per-device identity, survives IP changes, can't be shared across
    malicious peers), and falls back to the trusted-proxy client IP otherwise.
    """
    owner_hash = _community_owner_hash_from_request_optional(request)
    if owner_hash:
        return f"owner:{owner_hash}"
    return f"ip:{_client_ip(request)}"


def _enforce_rate_limit(request: Request, scope: str, *, limit: int, window_sec: float = RATE_LIMIT_WINDOW_SEC) -> None:
    _check_rate_limit_key(
        f"{scope}:{_rate_limit_identity_key(request)}",
        limit=limit,
        window_sec=window_sec,
    )


def _require_community_client_identity(request: Request) -> str:
    """
    Require the caller to identify themselves with an owner key. Returns the
    owner key hash. Raises 401 if the header is missing so anonymous bots can't
    hit expensive retrieval endpoints.

    Every shipped client (iOS + webapp) auto-generates and persists an owner
    key on first launch, so this is transparent to real users. The owner key
    is still a weak identifier — a determined attacker can generate arbitrary
    keys — but it closes casual API scraping and gives us a per-device bucket
    for rate limiting.
    """
    owner_hash = _community_owner_hash_from_request_optional(request)
    if not owner_hash:
        raise HTTPException(
            status_code=401,
            detail="Missing client identity header.",
        )
    return owner_hash


def _normalize_owner_key(raw_key: str | None) -> str:
    owner_key = str(raw_key or "").strip()
    if len(owner_key) < COMMUNITY_OWNER_KEY_MIN_LENGTH:
        raise HTTPException(status_code=400, detail="Missing or invalid community owner key.")
    return owner_key


def _community_owner_hash_from_request_optional(request: Request) -> str | None:
    raw_owner_key = str(request.headers.get(COMMUNITY_OWNER_HEADER) or "").strip()
    if not raw_owner_key:
        return None
    owner_key = _normalize_owner_key(raw_owner_key)
    return hashlib.sha256(owner_key.encode("utf-8")).hexdigest()


def _community_owner_hash_from_request(request: Request) -> str:
    owner_key_hash = _community_owner_hash_from_request_optional(request)
    if not owner_key_hash:
        raise HTTPException(status_code=400, detail="Missing or invalid community owner key.")
    return owner_key_hash


def _normalize_community_session_token(raw_token: str | None) -> str:
    token = str(raw_token or "").strip()
    if len(token) < COMMUNITY_SESSION_TOKEN_MIN_LENGTH:
        raise HTTPException(status_code=401, detail="Sign in to access your community sets.")
    return token


def _community_session_token_from_request(request: Request) -> str:
    return _normalize_community_session_token(request.headers.get(COMMUNITY_SESSION_HEADER))


def _normalize_community_username(raw_username: str) -> tuple[str, str]:
    username = str(raw_username or "").strip()
    normalized = username.lower()
    if not COMMUNITY_ACCOUNT_USERNAME_RE.fullmatch(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 characters and use only letters, numbers, dots, underscores, or hyphens.",
        )
    return username, normalized


def _normalize_community_password(raw_password: str) -> str:
    password = str(raw_password or "")
    if len(password) < 8 or len(password) > 128:
        raise HTTPException(status_code=400, detail="Password must be between 8 and 128 characters.")
    return password


def _normalize_community_email(raw_email: str) -> tuple[str, str]:
    email = str(raw_email or "").strip()
    _, parsed_email = parseaddr(email)
    normalized = parsed_email.strip().lower()
    if parsed_email != email or not COMMUNITY_ACCOUNT_EMAIL_RE.fullmatch(parsed_email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return parsed_email, normalized


def _hash_community_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        COMMUNITY_ACCOUNT_PASSWORD_ITERATIONS,
    ).hex()


def _verify_community_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    candidate = _hash_community_password(password, salt_hex)
    return secrets.compare_digest(candidate, str(expected_hash or ""))


def _community_session_expires_at_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=COMMUNITY_SESSION_TTL_DAYS)).isoformat()


def _community_verification_expires_at_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=COMMUNITY_VERIFICATION_TTL_MINUTES)).isoformat()


def _generate_community_verification_code() -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(COMMUNITY_VERIFICATION_CODE_LENGTH))


_local_verification_hmac_key_cache: str | None = None


def _load_or_create_local_verification_hmac_key() -> str:
    global _local_verification_hmac_key_cache
    if _local_verification_hmac_key_cache:
        return _local_verification_hmac_key_cache

    os.makedirs(settings.data_dir, exist_ok=True)
    secret_path = os.path.join(settings.data_dir, ".community_verification_hmac_key")
    try:
        with open(secret_path, "r", encoding="utf-8") as handle:
            existing_secret = handle.read().strip()
    except OSError:
        existing_secret = ""
    if existing_secret:
        _local_verification_hmac_key_cache = existing_secret
        return existing_secret

    generated_secret = secrets.token_urlsafe(48)
    try:
        with open(secret_path, "x", encoding="utf-8") as handle:
            handle.write(generated_secret)
        try:
            os.chmod(secret_path, 0o600)
        except OSError:
            pass
        _local_verification_hmac_key_cache = generated_secret
        return generated_secret
    except FileExistsError:
        try:
            with open(secret_path, "r", encoding="utf-8") as handle:
                concurrent_secret = handle.read().strip()
        except OSError:
            concurrent_secret = ""
        _local_verification_hmac_key_cache = concurrent_secret or generated_secret
        return _local_verification_hmac_key_cache
    except OSError:
        _local_verification_hmac_key_cache = generated_secret
        return generated_secret


def _community_verification_hmac_key() -> bytes:
    configured_secret = settings.verification_hmac_key.strip()
    if configured_secret:
        return configured_secret.encode("utf-8")
    if _is_hosted_runtime():
        raise HTTPException(status_code=503, detail="Account verification secret is not configured.")
    return _load_or_create_local_verification_hmac_key().encode("utf-8")


def _hash_community_verification_code(code: str) -> str:
    # Use HMAC with a server-side key so a DB leak doesn't expose the 1M-keyspace
    # 6-digit codes to trivial rainbow-table recovery.
    hmac_key = _community_verification_hmac_key()
    return hashlib.pbkdf2_hmac("sha256", code.encode("utf-8"), hmac_key, 100_000).hex()


def _parse_optional_iso_datetime(raw_value: object) -> datetime | None:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    normalized = raw.replace(" ", "T")
    if normalized.endswith(("z", "Z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _community_account_is_verified(row: dict[str, object]) -> bool:
    return _parse_optional_iso_datetime(row.get("verified_at")) is not None


def _community_email_verification_required() -> bool:
    return bool(settings.community_email_verification_required)


def _auto_verify_community_account_if_allowed(
    conn,
    request: Request,
    account_row: dict[str, object],
    *,
    claim_legacy_sets: bool = False,
) -> tuple[dict[str, object], int]:
    normalized_account = dict(account_row)
    if _community_email_verification_required() or _community_account_is_verified(normalized_account):
        return normalized_account, 0

    email = str(normalized_account.get("email") or "").strip()
    if not email:
        return normalized_account, 0

    account_id = str(normalized_account.get("id") or "").strip()
    if not account_id:
        return normalized_account, 0

    verified_at = now_iso()
    execute_modify(
        conn,
        """
        UPDATE community_accounts
        SET verified_at = ?, verification_code_hash = NULL, verification_expires_at = NULL, updated_at = ?
        WHERE id = ?
        """,
        (verified_at, verified_at, account_id),
    )
    normalized_account["verified_at"] = verified_at
    normalized_account["verification_code_hash"] = None
    normalized_account["verification_expires_at"] = None

    claimed_legacy_sets = 0
    if claim_legacy_sets:
        claimed_legacy_sets = _claim_legacy_community_sets_for_account(
            conn,
            request,
            account_id=account_id,
            legacy_claim_owner_key_hash=normalized_account.get("legacy_claim_owner_key_hash"),
        )
    return normalized_account, max(0, claimed_legacy_sets)


def _community_verification_debug_mode_enabled() -> bool:
    # Plaintext debug codes are allowed only for non-hosted runtimes.
    return _community_email_verification_required() and not _is_hosted_runtime()


def _community_verification_email_is_configured() -> bool:
    if settings.resend_api_key.strip() and settings.smtp_from_email.strip():
        return True
    return bool(settings.smtp_host.strip() and settings.smtp_from_email.strip())


def _require_hosted_verification_delivery_available() -> None:
    if not _community_email_verification_required():
        return
    if _community_verification_debug_mode_enabled():
        return
    if not _community_verification_email_is_configured():
        raise HTTPException(status_code=503, detail="Account verification email is not configured.")


def _warn_if_hosted_auth_email_is_unconfigured() -> None:
    if not _community_email_verification_required():
        return
    if not _is_hosted_runtime():
        return
    if _community_verification_email_is_configured():
        if settings.verification_hmac_key.strip():
            return
    if not _community_verification_email_is_configured():
        print(
            "Warning: hosted runtime detected but SMTP is not configured. "
            "Community account registration and unverified login will fail until SMTP_HOST and SMTP_FROM_EMAIL are set."
        )
    if not settings.verification_hmac_key.strip():
        print(
            "Warning: hosted runtime detected but VERIFICATION_HMAC_KEY is not configured. "
            "Community account verification will fail until a secret key is set."
        )


def _send_community_verification_email(*, email: str, username: str, code: str) -> None:
    greeting = f"@{username}" if str(username or "").strip() else "there"
    subject = "Verify your StudyReels account"
    text_body = (
        f"Hi {greeting},\n\n"
        f"Your StudyReels verification code is: {code}\n\n"
        f"This code expires in {COMMUNITY_VERIFICATION_TTL_MINUTES} minutes.\n"
        "If you did not create this account, you can ignore this email.\n"
    )
    # Spaced-out code for readability (e.g. "1 2 3 4 5 6")
    spaced_code = " ".join(code)
    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Verify your account</title>
</head>
<body style="margin:0;padding:0;background-color:#0f0f0f;font-family:Arial,Helvetica,sans-serif;color:#f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0f0f0f;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color:#1a1a1a;border-radius:8px;overflow:hidden;max-width:600px;width:100%;">
          <tr>
            <td style="padding:40px 48px 32px;">
              <h1 style="margin:0 0 8px;font-size:28px;font-weight:700;color:#ffffff;">Verify your account</h1>
              <p style="margin:0 0 24px;font-size:16px;color:#a0a0a0;">Enter this code to continue</p>
              <p style="margin:0 0 16px;font-size:16px;line-height:1.6;color:#e0e0e0;">
                Hi {greeting},
              </p>
              <p style="margin:0 0 24px;font-size:16px;line-height:1.6;color:#e0e0e0;">
                Here's your verification code:
              </p>
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 24px;">
                <tr>
                  <td align="center">
                    <div style="display:inline-block;background-color:#ffffff;border-radius:8px;padding:20px 40px;">
                      <span style="font-size:36px;font-weight:600;letter-spacing:10px;color:#0f0f0f;font-family:Arial,Helvetica,sans-serif;">{spaced_code}</span>
                    </div>
                  </td>
                </tr>
              </table>
              <p style="margin:0 0 32px;font-size:14px;line-height:1.6;color:#a0a0a0;">
                This code expires in {COMMUNITY_VERIFICATION_TTL_MINUTES} minutes.
              </p>
              <p style="margin:0;font-size:14px;color:#606060;">
                If you didn't request this code, you can safely ignore this email.
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding:24px 48px;border-top:1px solid #2a2a2a;">
              <p style="margin:0;font-size:13px;color:#606060;">
                &copy; StudyReels. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
    from_email = settings.smtp_from_email.strip()

    # Use Resend HTTP API if configured (avoids SMTP port blocking on PaaS).
    resend_api_key = settings.resend_api_key.strip()
    if resend_api_key:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {resend_api_key}"},
            json={"from": from_email, "to": [email], "subject": subject, "html": html_body, "text": text_body},
            timeout=15,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Resend API error {resp.status_code}: {resp.text}")
        return

    # Fallback: SMTP
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = email
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    smtp_host = settings.smtp_host.strip()
    smtp_port = max(1, int(settings.smtp_port))
    smtp_username = settings.smtp_username.strip()
    smtp_password = settings.smtp_password

    if settings.smtp_use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=15) as server:
            if smtp_username:
                server.login(smtp_username, smtp_password)
            server.send_message(message)
        return

    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
        if settings.smtp_use_tls:
            server.starttls()
        if smtp_username:
            server.login(smtp_username, smtp_password)
        server.send_message(message)


def _deliver_community_verification_code(*, email: str, username: str, code: str) -> str | None:
    if _community_verification_email_is_configured():
        try:
            _send_community_verification_email(email=email, username=username, code=code)
            return None
        except Exception as exc:
            import traceback
            print(f"[SMTP ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if _community_verification_debug_mode_enabled():
                return code
            raise HTTPException(status_code=502, detail="Could not send verification email. Try again.") from exc
    if _community_verification_debug_mode_enabled():
        return code
    raise HTTPException(status_code=503, detail="Account verification email is not configured.")


def _store_community_verification_code(conn, *, account_id: str) -> str:
    """Generate a verification code, store its hash, and return the plaintext code."""
    code = _generate_community_verification_code()
    execute_modify(
        conn,
        """
        UPDATE community_accounts
        SET verification_code_hash = ?, verification_expires_at = ?, updated_at = ?
        WHERE id = ?
        """,
        (_hash_community_verification_code(code), _community_verification_expires_at_iso(), now_iso(), account_id),
    )
    return code


def _issue_community_verification_code(conn, *, account_id: str, email: str, username: str) -> str | None:
    code = _store_community_verification_code(conn, account_id=account_id)
    return _deliver_community_verification_code(email=email, username=username, code=code)


def _community_signup_verification_id(*, owner_key_hash: str, email_normalized: str) -> str:
    return hashlib.sha256(f"{owner_key_hash}:{email_normalized}".encode("utf-8")).hexdigest()


def _load_community_signup_verification(conn, *, owner_key_hash: str, email_normalized: str) -> dict[str, object] | None:
    verification_id = _community_signup_verification_id(owner_key_hash=owner_key_hash, email_normalized=email_normalized)
    return fetch_one(
        conn,
        """
        SELECT
            id,
            owner_key_hash,
            email,
            email_normalized,
            verified_at,
            verification_code_hash,
            verification_expires_at,
            created_at,
            updated_at
        FROM community_signup_email_verifications
        WHERE id = ?
        LIMIT 1
        """,
        (verification_id,),
    )


def _community_signup_email_is_verified(row: dict[str, object] | None) -> bool:
    if not row:
        return False
    return _parse_optional_iso_datetime(row.get("verified_at")) is not None


def _store_community_signup_verification_code(
    conn,
    *,
    owner_key_hash: str,
    email: str,
    email_normalized: str,
) -> str:
    verification_id = _community_signup_verification_id(owner_key_hash=owner_key_hash, email_normalized=email_normalized)
    timestamp = now_iso()
    code = _generate_community_verification_code()
    code_hash = _hash_community_verification_code(code)
    expires_at = _community_verification_expires_at_iso()
    existing = _load_community_signup_verification(conn, owner_key_hash=owner_key_hash, email_normalized=email_normalized)
    if existing:
        execute_modify(
            conn,
            """
            UPDATE community_signup_email_verifications
            SET email = ?, email_normalized = ?, verified_at = NULL, verification_code_hash = ?, verification_expires_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (email, email_normalized, code_hash, expires_at, timestamp, verification_id),
        )
        return code
    insert(
        conn,
        "community_signup_email_verifications",
        {
            "id": verification_id,
            "owner_key_hash": owner_key_hash,
            "email": email,
            "email_normalized": email_normalized,
            "verified_at": None,
            "verification_code_hash": code_hash,
            "verification_expires_at": expires_at,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )
    return code


def _issue_community_signup_verification_code(
    conn,
    *,
    owner_key_hash: str,
    email: str,
    email_normalized: str,
    username: str | None,
) -> str | None:
    code = _store_community_signup_verification_code(
        conn,
        owner_key_hash=owner_key_hash,
        email=email,
        email_normalized=email_normalized,
    )
    delivery_username = str(username or "").strip()
    return _deliver_community_verification_code(email=email, username=delivery_username, code=code)


def _community_account_out(row: dict[str, object]) -> CommunityAccountOut:
    return CommunityAccountOut(
        id=str(row.get("id") or ""),
        username=str(row.get("username") or "").strip(),
        email=str(row.get("email") or "").strip() or None,
        is_verified=_community_account_is_verified(row),
    )


def _community_set_curator_for_account(account_row: dict[str, object]) -> str:
    return str(account_row.get("username") or "").strip() or "Community member"


def _create_community_session(conn, account_id: str) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    timestamp = now_iso()
    # Clean up expired sessions for this account.
    execute_modify(
        conn,
        "DELETE FROM community_sessions WHERE account_id = ? AND expires_at < ?",
        (account_id, timestamp),
    )
    # Enforce per-account session limit by removing oldest sessions.
    existing_sessions = fetch_all(
        conn,
        "SELECT id FROM community_sessions WHERE account_id = ? ORDER BY last_used_at DESC",
        (account_id,),
    )
    if len(existing_sessions) >= COMMUNITY_MAX_SESSIONS_PER_ACCOUNT:
        excess_ids = [str(s["id"]) for s in existing_sessions[COMMUNITY_MAX_SESSIONS_PER_ACCOUNT - 1:]]
        for excess_id in excess_ids:
            execute_modify(conn, "DELETE FROM community_sessions WHERE id = ?", (excess_id,))
    insert(
        conn,
        "community_sessions",
        {
            "id": str(uuid.uuid4()),
            "account_id": account_id,
            "token_hash": token_hash,
            "created_at": timestamp,
            "last_used_at": timestamp,
            "expires_at": _community_session_expires_at_iso(),
        },
    )
    return token


def _clear_legacy_claim_owner_key_hash(conn, *, account_id: str) -> None:
    execute_modify(
        conn,
        """
        UPDATE community_accounts
        SET legacy_claim_owner_key_hash = NULL, updated_at = ?
        WHERE id = ?
          AND legacy_claim_owner_key_hash IS NOT NULL
        """,
        (now_iso(), account_id),
    )


def _claim_legacy_community_sets_for_account(
    conn,
    request: Request,
    *,
    account_id: str,
    legacy_claim_owner_key_hash: object,
) -> int:
    stored_owner_key_hash = str(legacy_claim_owner_key_hash or "").strip()
    if not stored_owner_key_hash or stored_owner_key_hash == LEGACY_COMMUNITY_OWNER_HASH:
        return 0
    request_owner_key_hash = _community_owner_hash_from_request_optional(request)
    if request_owner_key_hash != stored_owner_key_hash:
        return 0
    claimed_count = execute_modify(
        conn,
        """
        UPDATE community_sets
        SET owner_account_id = ?, visibility = ?
        WHERE owner_account_id IS NULL
          AND owner_key_hash = ?
          AND owner_key_hash <> ?
        """,
        (account_id, DEFAULT_COMMUNITY_VISIBILITY, stored_owner_key_hash, LEGACY_COMMUNITY_OWNER_HASH),
    )
    _clear_legacy_claim_owner_key_hash(conn, account_id=account_id)
    return claimed_count


def _require_authenticated_community_account(conn, request: Request) -> dict[str, object]:
    session_token = _community_session_token_from_request(request)
    session_token_hash = hashlib.sha256(session_token.encode("utf-8")).hexdigest()
    row = fetch_one(
        conn,
        """
        SELECT
            a.id,
            a.username,
            a.email,
            a.username_normalized,
            a.verified_at,
            s.id AS session_id,
            s.expires_at
        FROM community_sessions AS s
        JOIN community_accounts AS a ON a.id = s.account_id
        WHERE s.token_hash = ?
        LIMIT 1
        """,
        (session_token_hash,),
    )
    if not row:
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")
    expires_at = _parse_optional_iso_datetime(row.get("expires_at"))
    now_dt = datetime.now(timezone.utc)
    if expires_at is None or expires_at <= now_dt:
        execute_modify(conn, "DELETE FROM community_sessions WHERE id = ?", (row["session_id"],))
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")
    execute_modify(
        conn,
        "UPDATE community_sessions SET last_used_at = ?, expires_at = ? WHERE id = ?",
        (now_iso(), _community_session_expires_at_iso(), row["session_id"]),
    )
    return row


def _require_verified_community_account(conn, request: Request) -> dict[str, object]:
    account = _require_authenticated_community_account(conn, request)
    account, _ = _auto_verify_community_account_if_allowed(conn, request, account, claim_legacy_sets=True)
    if not _community_account_is_verified(account):
        raise HTTPException(status_code=403, detail="Verify your account to access your sets.")
    return account


def _require_verified_provider_account(
    conn: Any,
    request: Request,
) -> dict[str, object]:
    """Apply the typed account gate to billing and newly-created provider work."""
    try:
        return _require_verified_community_account(conn, request)
    except HTTPException as exc:
        message = (
            "Sign in to a verified ReelAI account to start a new search."
            if exc.status_code == 401
            else "Verify your ReelAI account to start a new search."
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": "verified_account_required", "message": message},
        ) from exc


def _reserve_search_or_http(conn: Any, **kwargs: Any) -> dict[str, Any] | None:
    try:
        return reserve_search(conn, **kwargs)
    except DailySearchLimitReached as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.detail(),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


def _begin_sync_search_quota(
    request: Request,
    *,
    surface: str,
    request_fingerprint: str,
    material_id: str | None = None,
    charge_search: bool = True,
) -> dict[str, Any] | None:
    with get_conn(transactional=True) as conn:
        account = _require_verified_provider_account(conn, request)
        account_id = str(account["id"])
        enforcement_enabled = billing_enforcement_enabled()
        try:
            idempotency_key = normalize_idempotency_key(
                request.headers.get("Idempotency-Key")
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not idempotency_key:
            if enforcement_enabled:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "idempotency_key_required",
                        "message": "Idempotency-Key is required for a new search.",
                    },
                )
            return None
        operation_hash = hashlib.sha256(
            f"{surface}:{idempotency_key}".encode("utf-8")
        ).hexdigest()
        operation_key = f"{surface}:{operation_hash}"
        try:
            idempotency = reserve_idempotency_key(
                conn,
                scope=f"billing:{surface}",
                learner_id=account_id,
                raw_key=idempotency_key,
                fingerprint=request_fingerprint,
                resource_id=operation_key,
                stale_after_seconds=2 * 60 * 60,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "idempotency_key_reused", "message": str(exc)},
            ) from exc
        if not idempotency.owner:
            if idempotency.status == "completed" and idempotency.response is not None:
                return {"replay_response": idempotency.response}
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "idempotency_in_progress",
                    "message": "This search is already being processed.",
                },
                headers={"Retry-After": "1"},
            )
        if enforcement_enabled and charge_search:
            _reserve_search_or_http(
                conn,
                account_id=account_id,
                operation_key=operation_key,
                surface=surface,
                material_id=material_id,
            )
    return {
        "account_id": account_id,
        "operation_key": operation_key,
        "idempotency_key": idempotency_key,
        "attempt_token": str(idempotency.attempt_token or ""),
    }


def _settle_sync_search_quota(
    quota: dict[str, Any] | None,
    *,
    usable_result: bool,
    response: dict[str, Any] | None = None,
) -> None:
    if quota is None or quota.get("replay_response") is not None:
        return
    with get_conn(transactional=True) as conn:
        settle_operation(
            conn,
            account_id=str(quota["account_id"]),
            operation_key=str(quota["operation_key"]),
            usable_result=usable_result,
        )
        scope = f"billing:{str(quota['operation_key']).split(':', 1)[0]}"
        if response is not None:
            completed = complete_idempotency_key(
                conn,
                scope=scope,
                learner_id=str(quota["account_id"]),
                raw_key=str(quota["idempotency_key"]),
                resource_id=str(quota["operation_key"]),
                attempt_token=str(quota["attempt_token"]),
                response=response,
            )
            if not completed:
                raise RuntimeError("search idempotency reservation was lost")
        else:
            released = release_idempotency_key(
                conn,
                scope=scope,
                learner_id=str(quota["account_id"]),
                raw_key=str(quota["idempotency_key"]),
                resource_id=str(quota["operation_key"]),
                attempt_token=str(quota["attempt_token"]),
            )
            if not released:
                raise RuntimeError("search idempotency reservation was lost")


def _require_community_set_owner_access(
    stored_owner_account_id: object,
    account_id: str,
    *,
    action: str,
) -> str:
    normalized_owner_account_id = str(stored_owner_account_id or "").strip()
    if not normalized_owner_account_id:
        raise HTTPException(
            status_code=403,
            detail=f"This community set is not attached to an account yet and cannot be {action} automatically.",
        )
    if normalized_owner_account_id != account_id:
        raise HTTPException(status_code=403, detail=f"You do not have permission to {action} this community set.")
    return normalized_owner_account_id


def _validate_community_reel_urls(
    *,
    platform: Literal["youtube", "instagram", "tiktok"],
    source_url: str,
    embed_url: str,
) -> tuple[str, str]:
    safe_source = _normalize_public_http_url(source_url, "source_url")
    safe_embed = _normalize_public_http_url(embed_url, "embed_url")
    source_parsed = urlparse(safe_source)
    embed_parsed = urlparse(safe_embed)
    source_host = (source_parsed.hostname or "").lower()
    embed_host = (embed_parsed.hostname or "").lower()
    embed_path = (embed_parsed.path or "").lower()

    if platform == "youtube":
        if not (_host_matches(source_host, "youtube.com") or source_host == "youtu.be"):
            raise HTTPException(status_code=400, detail="source_url host must be a YouTube URL for platform=youtube.")
        if not _host_matches(embed_host, "youtube.com") or not embed_path.startswith("/embed/"):
            raise HTTPException(status_code=400, detail="embed_url must be a YouTube embed URL for platform=youtube.")
        return safe_source, safe_embed

    if platform == "instagram":
        if not _host_matches(source_host, "instagram.com"):
            raise HTTPException(status_code=400, detail="source_url host must be an Instagram URL for platform=instagram.")
        if not _host_matches(embed_host, "instagram.com") or not re.search(r"/embed(?:/|$)", embed_path):
            raise HTTPException(status_code=400, detail="embed_url must be an Instagram embed URL for platform=instagram.")
        return safe_source, safe_embed

    if not _host_matches(source_host, "tiktok.com"):
        raise HTTPException(status_code=400, detail="source_url host must be a TikTok URL for platform=tiktok.")
    if not _host_matches(embed_host, "tiktok.com") or "/embed/" not in embed_path:
        raise HTTPException(status_code=400, detail="embed_url must be a TikTok embed URL for platform=tiktok.")
    return safe_source, safe_embed


def _community_duration_source_host_allowed(host: str) -> bool:
    return bool(
        _host_matches(host, "youtube.com")
        or host == "youtu.be"
        or _host_matches(host, "instagram.com")
        or _host_matches(host, "tiktok.com")
    )


def _normalize_community_duration_source_url(raw_url: str) -> str:
    normalized = _normalize_public_http_url(raw_url, "source_url")
    host = (urlparse(normalized).hostname or "").lower()
    if not _community_duration_source_host_allowed(host):
        raise HTTPException(
            status_code=400,
            detail="source_url must be a supported YouTube, Instagram, or TikTok URL.",
        )
    return normalized


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


def _parse_iso8601_duration(value: str) -> int:
    match = re.match(
        r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$",
        str(value or ""),
    )
    if not match:
        return 0
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _extract_duration_from_html(html: str) -> float | None:
    """Pull a video's duration (in seconds) out of raw HTML.

    The function tries several strategies in order of how reliable they are.
    Each strategy returns as soon as it finds a match, so adding more of
    them costs nothing for the common case. Strategies are ordered so the
    ones that *usually* work first come first — this keeps the latency low.

    Strategies (in order):
      1. OpenGraph `<meta property="video:duration" content="...">` — the
         most portable format, used by YouTube, Twitter, Facebook, etc.
      2. Schema.org `itemprop="duration"` — used by sites that follow the
         microdata spec.
      3. YouTube internal `"lengthSeconds":"..."` — YouTube-specific but
         extremely reliable when present.
      4. JSON-LD `"duration":"PT..."` — ISO-8601 durations embedded in
         <script type="application/ld+json"> blocks. Schema.org standard.
      5. Instagram / TikTok `"video_duration":N` (and variants) — these
         platforms don't expose OpenGraph durations consistently, but
         their SSR payloads include this field.
      6. Generic `"duration": <number>` — last-resort, accepts anything
         numeric labelled "duration" inside a JSON blob.
    """
    if not html:
        return None

    # -- Strategy 1: OpenGraph meta tag -----------------------------------
    # The `content` attribute can appear before or after the `property`
    # attribute, and the spec allows either single or double quotes, so we
    # keep four variants rather than hand-writing one monster regex.
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

    # -- Strategy 2: Schema.org microdata ---------------------------------
    # Used by sites that publish structured data directly in the DOM.
    itemprop_match = re.search(
        r'itemprop=["\']duration["\'][^>]*content=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if itemprop_match:
        raw = itemprop_match.group(1).strip()
        # Could be either an ISO-8601 duration ("PT1M30S") or a plain number.
        if raw.upper().startswith("P"):
            iso_seconds = _parse_iso8601_duration(raw)
            duration = _normalize_duration_seconds(iso_seconds)
            if duration is not None:
                return duration
        else:
            duration = _normalize_duration_seconds(raw)
            if duration is not None:
                return duration

    # -- Strategy 3: YouTube-specific lengthSeconds -----------------------
    # Embedded in YouTube's player config; extremely reliable when it's
    # present. The value is always a quoted integer string.
    length_seconds_match = re.search(r'"lengthSeconds"\s*:\s*"(\d{1,7})"', html)
    if length_seconds_match:
        duration = _normalize_duration_seconds(length_seconds_match.group(1))
        if duration is not None:
            return duration

    # -- Strategy 4: JSON-LD ISO-8601 duration ----------------------------
    # Schema.org says `duration` on a VideoObject is an ISO-8601 string
    # starting with "P" (e.g. "PT1M30S" = 1 minute 30 seconds).
    json_ld_duration = re.search(r'"duration"\s*:\s*"((?:P|PT)[^"]+)"', html, flags=re.IGNORECASE)
    if json_ld_duration:
        iso_duration = _parse_iso8601_duration(json_ld_duration.group(1))
        duration = _normalize_duration_seconds(iso_duration)
        if duration is not None:
            return duration

    # -- Strategy 5: Instagram / TikTok SSR payloads ----------------------
    # Instagram uses `"video_duration":N` (a float, in seconds) and TikTok
    # sometimes exposes `"duration":N` on a video object inside its __NEXT_DATA__
    # blob. Both are unquoted numbers, distinct from the JSON-LD variant.
    ig_tt_patterns = (
        r'"video_duration"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r'"videoDuration"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        # TikTok SIGI_STATE often uses "duration" (in seconds) inside a
        # videoObject — same key as Strategy 6 but we match it here first
        # when it looks like an integer of seconds, not ISO-8601.
    )
    for pattern in ig_tt_patterns:
        match = re.search(pattern, html)
        if not match:
            continue
        duration = _normalize_duration_seconds(match.group(1))
        if duration is not None:
            return duration

    # -- Strategy 6: Generic `"duration": <number>` ------------------------
    # Catch-all for any unknown JSON payload that exposes a numeric duration
    # labelled plainly. We run this LAST because the word "duration" is
    # common and could collide with non-video fields earlier in the page.
    generic_numeric_duration = re.search(r'"duration"\s*:\s*([0-9]{1,7}(?:\.[0-9]+)?)', html)
    if generic_numeric_duration:
        duration = _normalize_duration_seconds(generic_numeric_duration.group(1))
        if duration is not None:
            return duration

    return None


def _fetch_duration_from_source_page(source_url: str) -> float | None:
    """Fetch a source URL and try to extract the video duration from the HTML.

    Why manual redirect handling?
      We don't let requests follow redirects automatically because each new
      location has to pass our host allow-list check (YouTube, Instagram,
      TikTok only). Blindly following redirects would let an attacker feed
      us a `bit.ly` or shortener URL that redirects to an internal service.
      Manually stepping through redirects lets us re-validate every hop.

    Edge cases we handle:
      - Network/transport failure → log and return None (caller displays
        a generic "couldn't determine duration" message, not a crash)
      - Missing Location header on a 3xx → treat as a dead end
      - Redirect that lands on a disallowed host → reject it
      - Rate-limit (429) / server error (5xx) → log and return None so the
        caller knows not to retry aggressively
      - Oversized HTML → we slice to 900KB which is more than enough for
        any <head> + initial JSON payload, and protects against huge pages
    """
    current_url = source_url
    for hop in range(MAX_DURATION_FETCH_REDIRECTS + 1):
        try:
            response = requests.get(
                current_url,
                allow_redirects=False,
                timeout=COMMUNITY_REEL_DURATION_TIMEOUT_SEC,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        except requests.Timeout:
            logger.info(
                "Duration fetch timed out after %.1fs: url=%s hop=%d",
                COMMUNITY_REEL_DURATION_TIMEOUT_SEC,
                current_url,
                hop,
            )
            return None
        except requests.RequestException as exc:
            logger.info(
                "Duration fetch transport error: url=%s hop=%d type=%s error=%s",
                current_url,
                hop,
                type(exc).__name__,
                exc,
            )
            return None

        status = response.status_code

        # Handle 3xx manually so we can re-validate the redirect target
        # against the host allow-list.
        if 300 <= status < 400:
            location = str(response.headers.get("Location") or "").strip()
            if not location:
                logger.debug("Duration fetch got %d but no Location header: url=%s", status, current_url)
                return None
            try:
                current_url = _normalize_community_duration_source_url(urljoin(current_url, location))
            except HTTPException as exc:
                # Host allow-list rejected the redirect target. That's not
                # a crash — it's the safety valve doing its job.
                logger.info(
                    "Duration fetch redirect rejected: url=%s location=%s reason=%s",
                    current_url,
                    location,
                    exc.detail,
                )
                return None
            continue

        # Explicit handling for common failure modes, each with its own log
        # line so you can tell them apart in production.
        if status == 404:
            logger.debug("Duration fetch got 404 (video likely deleted/private): url=%s", current_url)
            return None
        if status == 429:
            logger.warning("Duration fetch was rate-limited (429): url=%s", current_url)
            return None
        if 500 <= status < 600:
            logger.info("Duration fetch got server error %d: url=%s", status, current_url)
            return None
        if status >= 400:
            logger.debug("Duration fetch got HTTP %d: url=%s", status, current_url)
            return None

        # Some platforms (Instagram especially) return a tiny skeleton
        # response when the content is private or deleted — the status is
        # 200 but the body has no useful data. We still try to extract,
        # but log a debug hint if the body is suspiciously small.
        body_text = response.text or ""
        if len(body_text) < 1000:
            logger.debug(
                "Duration fetch body is very small (%d bytes) — may be private or removed: url=%s",
                len(body_text),
                current_url,
            )
        html = body_text[:900_000]
        duration = _extract_duration_from_html(html)
        if duration is None:
            logger.debug(
                "Duration extraction returned no value for url=%s (html_bytes=%d). "
                "Add a new pattern to _extract_duration_from_html if this should have worked.",
                current_url,
                len(html),
            )
        return duration
    # Exceeded redirect budget.
    logger.info("Duration fetch exceeded redirect budget: original_url=%s", source_url)
    return None


def _resolve_community_reel_duration_sec(source_url: str) -> float | None:
    normalized_source_url = _normalize_community_duration_source_url(source_url)
    parsed = urlparse(normalized_source_url)
    host = (parsed.hostname or "").lower()

    if "youtube.com" in host or "youtu.be" in host:
        try:
            artifact = fetch_transcript_artifact(
                normalized_source_url,
                deadline_monotonic=time.monotonic() + 30.0,
            )
        except ClipEngineProviderError:
            return None
        return _normalize_duration_seconds(artifact.duration_sec)

    return _fetch_duration_from_source_page(normalized_source_url)


def _cached_community_reel_duration_sec(source_url: str) -> tuple[bool, float | None]:
    """Return a cached YouTube duration without allowing provider or page I/O."""
    normalized_source_url = _normalize_community_duration_source_url(source_url)
    parsed = urlparse(normalized_source_url)
    host = (parsed.hostname or "").lower()
    if "youtube.com" not in host and host != "youtu.be":
        return False, None
    video_id = normalize_youtube_video_id(normalized_source_url)
    if video_id is None:
        return False, None
    store = DatabaseProviderCache()
    for native_mode in (True, False):
        artifact = store.get_transcript(
            video_id=video_id,
            provider="supadata",
            requested_language="en",
            native_mode=native_mode,
            schema_version=TRANSCRIPT_SCHEMA_VERSION,
        )
        if artifact is not None:
            return True, _normalize_duration_seconds(artifact.duration_sec)
    return False, None


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


def _normalize_optional_community_reel_id(raw_value: object) -> str | None:
    reel_id = str(raw_value or "").strip()
    if not reel_id:
        return None
    return reel_id


def _normalize_min_relevance(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(-1.0, min(1.2, parsed))


def _normalize_preferred_video_duration(value: str | None) -> Literal["any", "short", "medium", "long"]:
    if value in VALID_VIDEO_DURATION_PREFS:
        return value
    return "any"


def _normalize_default_input_mode(value: str | None) -> Literal["topic", "source", "file"]:
    if value in VALID_SEARCH_INPUT_MODES:
        return value  # type: ignore[return-value]
    return DEFAULT_SETTINGS_DEFAULT_INPUT_MODE  # type: ignore[return-value]


def _normalize_settings_min_relevance_threshold(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_SETTINGS_MIN_RELEVANCE_THRESHOLD
    safe = max(MIN_SETTINGS_MIN_RELEVANCE_THRESHOLD, min(MAX_SETTINGS_MIN_RELEVANCE_THRESHOLD, parsed))
    return round(safe, 2)


def _normalize_settings_start_muted(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(_to_int(value, 1))


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
    default_min = max(MIN_TARGET_CLIP_DURATION_SEC, int(round(safe_target * 0.35)))
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


def _serialize_community_settings(row: dict | None) -> CommunitySettingsResponse:
    source = row or {}
    generation_mode_raw = str(source.get("generation_mode") or "").strip().lower()
    generation_mode = generation_mode_raw if generation_mode_raw in {"slow", "fast"} else "slow"
    default_input_mode = _normalize_default_input_mode(str(source.get("default_input_mode") or "").strip().lower() or None)
    min_relevance_threshold = _normalize_settings_min_relevance_threshold(source.get("min_relevance_threshold"))
    start_muted = _normalize_settings_start_muted(source.get("start_muted", 1 if DEFAULT_SETTINGS_START_MUTED else 0))
    preferred_video_duration = _normalize_preferred_video_duration(
        str(source.get("preferred_video_duration") or "").strip().lower() or None
    )
    target_clip_duration_sec, target_clip_duration_min_sec, target_clip_duration_max_sec = _resolve_target_clip_duration_bounds(
        source.get("target_clip_duration_sec", DEFAULT_TARGET_CLIP_DURATION_SEC),
        source.get("target_clip_duration_min_sec", DEFAULT_SETTINGS_TARGET_CLIP_DURATION_MIN_SEC),
        source.get("target_clip_duration_max_sec", DEFAULT_SETTINGS_TARGET_CLIP_DURATION_MAX_SEC),
    )
    autoplay_next_reel = bool(source.get("autoplay_next_reel", False))
    return CommunitySettingsResponse(
        generation_mode=generation_mode,  # type: ignore[arg-type]
        default_input_mode=default_input_mode,
        min_relevance_threshold=min_relevance_threshold,
        start_muted=start_muted,
        creative_commons_only=bool(source.get("creative_commons_only", False)),
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        autoplay_next_reel=autoplay_next_reel,
    )


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


def _reel_relevance_value(reel: dict[str, Any]) -> float | None:
    relevance = reel.get("relevance_score")
    if not isinstance(relevance, (int, float)):
        return None
    return float(relevance)


def _adaptive_min_relevance_floor(reels: list[dict[str, Any]], min_relevance: float | None) -> float | None:
    if min_relevance is None:
        return None
    numeric_scores = sorted(
        [relevance for reel in reels if (relevance := _reel_relevance_value(reel)) is not None],
        reverse=True,
    )
    if not numeric_scores:
        return None
    best_score = float(numeric_scores[0])
    if best_score >= min_relevance:
        return float(min_relevance)
    fallback_window = max(1, min(6, len(numeric_scores)))
    cluster_floor = float(numeric_scores[fallback_window - 1])
    return max(0.08, cluster_floor, best_score * 0.72)


def _top_relevance_fallback_reels(reels: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    ranked: list[tuple[float, int]] = []
    for index, reel in enumerate(reels):
        relevance = _reel_relevance_value(reel)
        if relevance is None:
            continue
        ranked.append((relevance, index))
    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    keep_indexes = {index for _score, index in ranked[: max(1, limit)]}
    return [reel for index, reel in enumerate(reels) if index in keep_indexes]


def _filter_reels_by_min_relevance(reels: list[dict], min_relevance: float | None) -> list[dict]:
    if min_relevance is None:
        return reels
    filtered: list[dict] = []
    for reel in reels:
        relevance = _reel_relevance_value(reel)
        if relevance is not None and relevance < min_relevance:
            continue
        filtered.append(reel)
    if filtered:
        return filtered

    adaptive_floor = _adaptive_min_relevance_floor(reels, min_relevance)
    if adaptive_floor is None or adaptive_floor >= min_relevance:
        return filtered

    relaxed: list[dict] = []
    for reel in reels:
        relevance = _reel_relevance_value(reel)
        if relevance is not None and relevance < adaptive_floor:
            continue
        relaxed.append(reel)
    if relaxed:
        return relaxed
    return _top_relevance_fallback_reels(reels)


def _filter_reels_by_video_preferences(
    reels: list[dict],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None = None,
    target_clip_duration_max_sec: int | None = None,
) -> list[dict]:
    safe_duration_pref = _normalize_preferred_video_duration(preferred_video_duration)
    filtered: list[dict[str, Any]] = []
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

        normalized = dict(reel)
        if clip_duration_value > 0:
            normalized["clip_duration_sec"] = round(clip_duration_value, 2)
        normalized["duration_preference_met"] = True
        normalized["duration_fit"] = "in_range"
        filtered.append(normalized)
    return filtered


def _generation_content_fingerprint(
    conn,
    *,
    material_id: str,
    concept_id: str | None,
) -> str:
    try:
        material = fetch_one(
            conn,
            "SELECT subject_tag, raw_text, source_type FROM materials WHERE id = ?",
            (material_id,),
        ) or {}
    except Exception as exc:
        if "column" not in str(exc).lower():
            raise
        material = fetch_one(conn, "SELECT * FROM materials WHERE id = ?", (material_id,)) or {}
    params: tuple[Any, ...] = (material_id,)
    concept_where = "material_id = ?"
    if concept_id:
        concept_where += " AND id = ?"
        params = (material_id, concept_id)
    try:
        concepts = fetch_all(
            conn,
            "SELECT id, title, keywords_json, summary FROM concepts "
            f"WHERE {concept_where} ORDER BY id",
            params,
        )
    except Exception as exc:
        if "column" not in str(exc).lower() and "no such table" not in str(exc).lower():
            raise
        try:
            concepts = fetch_all(
                conn,
                f"SELECT * FROM concepts WHERE {concept_where} ORDER BY id",
                params,
            )
        except Exception as fallback_exc:
            if "no such table" not in str(fallback_exc).lower():
                raise
            concepts = []
    payload = {
        "material": {
            "id": material_id,
            "subject_tag": str(material.get("subject_tag") or ""),
            "raw_text": str(material.get("raw_text") or ""),
            "source_type": str(material.get("source_type") or ""),
        },
        "concepts": concepts,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_generation_head_id(material_id: str, request_key: str) -> str:
    return hashlib.sha256(f"{material_id}:{request_key}".encode("utf-8")).hexdigest()


def _serialize_reels_for_request(
    reels: list[dict[str, Any]],
    *,
    min_relevance: float | None,
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
) -> list[dict[str, Any]]:
    filtered = _filter_reels_by_min_relevance(reels, min_relevance)
    filtered = _filter_reels_by_video_preferences(
        filtered,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
    )
    return filtered


def _dedupe_request_terms(raw_terms: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = " ".join(str(term or "").lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(term).strip())
    return deduped


def _request_root_anchor_terms(subject_tag: str | None) -> tuple[list[str], list[str], list[str]]:
    cleaned = str(subject_tag or "").strip()
    if not cleaned:
        return ([], [], [])
    try:
        with get_conn() as conn:
            plan = build_search_query_plan(conn, literal_query=cleaned)
    except Exception:
        plan = None
    if plan is None or plan.ai_status != "validated":
        return ([cleaned], [], [])

    roots = _dedupe_request_terms(
        [cleaned, plan.canonical_query, *plan.accepted_aliases]
    )
    companions = _dedupe_request_terms(
        [*plan.accepted_subtopics, *plan.accepted_related_terms]
    )
    return roots, companions, companions[:10]


def _request_reel_text(reel: dict[str, Any]) -> str:
    parts: list[str] = [
        str(reel.get("video_title") or ""),
        str(reel.get("video_description") or ""),
        str(reel.get("transcript_snippet") or ""),
    ]
    matched_terms = reel.get("matched_terms") or []
    if isinstance(matched_terms, list):
        parts.extend(str(item or "") for item in matched_terms[:8])
    return " ".join(parts).strip().lower()


def _request_text_has_anchor(text: str, terms: list[str]) -> bool:
    lowered = f" {semantic_key(text)} "
    text_tokens = set(semantic_tokens(text))
    for raw_term in terms:
        normalized = semantic_key(raw_term)
        if not normalized:
            continue
        if f" {normalized} " in lowered:
            return True
        tokens = normalized.split()
        if len(tokens) > 1 and set(tokens).issubset(text_tokens):
            return True
    return False


def _request_matched_anchor_terms(text: str, terms: list[str]) -> list[str]:
    lowered = f" {semantic_key(text)} "
    text_tokens = set(semantic_tokens(text))
    matches: list[str] = []
    seen: set[str] = set()
    for raw_term in terms:
        candidate = str(raw_term or "").strip()
        normalized = semantic_key(candidate)
        if not candidate or not normalized or normalized in seen:
            continue
        if f" {normalized} " in lowered:
            matches.append(candidate)
            seen.add(normalized)
            continue
        tokens = normalized.split()
        if len(tokens) > 1 and set(tokens).issubset(text_tokens):
            matches.append(candidate)
            seen.add(normalized)
    matches.sort(key=lambda item: (-len(str(item)), str(item).lower()))
    return matches


def _request_similarity_profile(subject_tag: str | None) -> dict[str, float]:
    breadth_class = reel_service._topic_breadth_class(subject_tag)
    novelty_profile = reel_service._topic_novelty_profile(
        subject_tag=subject_tag,
        retrieval_profile="deep",
        fast_mode=False,
    )
    return {
        "cross_video_similarity": float(novelty_profile.get("cross_video_similarity") or 0.9),
        "same_video_similarity": float(novelty_profile.get("same_video_similarity") or 0.86),
        "breadth_class": breadth_class,
    }


def _request_reel_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_text = _request_reel_text(left)
    right_text = _request_reel_text(right)
    left_tokens = set(semantic_tokens(left_text))
    right_tokens = set(semantic_tokens(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return float(overlap / max(1, union))


def _request_source_surface_allowed(
    reel: dict[str, Any],
    *,
    page: int,
) -> bool:
    surface = str(reel.get("source_surface") or "").strip().lower()
    if page <= 1 and surface in {
        "youtube_related",
        "youtube_channel",
        "duckduckgo_site",
        "duckduckgo_quoted",
        "bing_site",
        "bing_quoted",
        "local_cache",
    }:
        return False
    if page <= 2 and surface in {
        "youtube_related",
        "youtube_channel",
        "duckduckgo_site",
        "duckduckgo_quoted",
        "bing_site",
        "bing_quoted",
        "local_cache",
    }:
        return False
    if page <= 3 and surface in {
        "duckduckgo_site",
        "duckduckgo_quoted",
        "bing_site",
        "bing_quoted",
        "local_cache",
    }:
        return False
    return True


def _request_rank_score(
    reel: dict[str, Any],
    *,
    page: int,
    has_root_anchor: bool,
    has_companion_anchor: bool,
    has_page_one_companion_anchor: bool,
    educational_support: bool,
) -> float:
    relevance_score = float(reel.get("relevance_score") or 0.0)
    matched_terms = reel.get("matched_terms") or []
    if not isinstance(matched_terms, list):
        matched_terms = []
    precision_signal = min(1.0, 0.22 * len([term for term in matched_terms if str(term or "").strip()]))
    exactness = 1.0 if has_root_anchor else 0.74 if has_page_one_companion_anchor else 0.58 if has_companion_anchor else 0.0
    source_surface = str(reel.get("source_surface") or "").strip().lower()
    source_trust = {
        "youtube_api": 1.0,
        "youtube_html": 0.96,
        "local_bootstrap": 0.82,
        "youtube_related": 0.72,
        "youtube_channel": 0.58,
        "duckduckgo_site": 0.24,
        "duckduckgo_quoted": 0.28,
        "bing_site": 0.24,
        "bing_quoted": 0.28,
        "local_cache": 0.2,
    }.get(source_surface, 0.5)
    educational = 1.0 if educational_support else 0.0
    if page <= 2:
        return (
            0.35 * exactness
            + 0.25 * relevance_score
            + 0.2 * precision_signal
            + 0.1 * source_trust
            + 0.05 * educational
            + 0.05 * max(0.0, float(reel.get("score") or 0.0))
            + _request_stage_bonus(reel, page=page)
        )
    if page <= 5:
        return (
            0.25 * exactness
            + 0.2 * relevance_score
            + 0.15 * precision_signal
            + 0.15 * source_trust
            + 0.15 * educational
            + 0.1 * max(0.0, float(reel.get("score") or 0.0))
            + _request_stage_bonus(reel, page=page)
        )
    return (
        0.2 * exactness
        + 0.2 * relevance_score
        + 0.2 * source_trust
        + 0.15 * precision_signal
        + 0.15 * educational
        + 0.1 * max(0.0, float(reel.get("score") or 0.0))
        + _request_stage_bonus(reel, page=page)
    )


def _request_diversity_ready(
    *,
    per_video_counts: dict[str, int],
    per_anchor_counts: dict[str, int],
) -> bool:
    represented_videos = sum(1 for count in per_video_counts.values() if count > 0)
    represented_anchors = sum(1 for count in per_anchor_counts.values() if count > 0)
    return represented_videos >= 4 and represented_anchors >= 3


def _request_stage_bonus(reel: dict[str, Any], *, page: int) -> float:
    source_surface = str(reel.get("source_surface") or "").strip().lower()
    retrieval_stage = str(reel.get("retrieval_stage") or "").strip().lower()
    query_strategy = str(reel.get("query_strategy") or "").strip().lower()

    if page <= 2:
        source_bonus = {
            "youtube_api": 0.18,
            "youtube_html": 0.16,
            "local_bootstrap": 0.12,
            "youtube_related": -0.08,
            "youtube_channel": -0.12,
            "duckduckgo_site": -0.12,
            "bing_site": -0.12,
            "duckduckgo_quoted": -0.1,
            "bing_quoted": -0.1,
            "local_cache": -0.16,
        }.get(source_surface, 0.0)
        stage_bonus = {"high_precision": 0.09, "broad": 0.02, "recovery": -0.2}.get(retrieval_stage, 0.0)
        strategy_bonus = {
            "literal": 0.05,
            "explained": 0.04,
            "tutorial": 0.04,
            "worked_example": 0.05,
            "recovery_adjacent": -0.18,
        }.get(query_strategy, 0.0)
    elif page <= 5:
        source_bonus = {
            "youtube_api": 0.08,
            "youtube_html": 0.08,
            "local_bootstrap": 0.06,
            "youtube_related": 0.04,
            "youtube_channel": 0.0,
            "duckduckgo_site": -0.04,
            "bing_site": -0.04,
            "duckduckgo_quoted": -0.02,
            "bing_quoted": -0.02,
            "local_cache": -0.06,
        }.get(source_surface, 0.0)
        stage_bonus = {"high_precision": 0.04, "broad": 0.02, "recovery": -0.04}.get(retrieval_stage, 0.0)
        strategy_bonus = {"recovery_adjacent": -0.03}.get(query_strategy, 0.0)
    else:
        source_bonus = {
            "youtube_api": 0.04,
            "youtube_html": 0.04,
            "local_bootstrap": 0.03,
            "youtube_related": 0.03,
            "youtube_channel": 0.01,
            "duckduckgo_site": -0.02,
            "bing_site": -0.02,
            "duckduckgo_quoted": -0.01,
            "bing_quoted": -0.01,
            "local_cache": -0.02,
        }.get(source_surface, 0.0)
        stage_bonus = {"high_precision": 0.02, "broad": 0.02, "recovery": -0.01}.get(retrieval_stage, 0.0)
        strategy_bonus = {"recovery_adjacent": -0.01}.get(query_strategy, 0.0)
    return source_bonus + stage_bonus + strategy_bonus


def _request_page_video_cap(reel: dict[str, Any], *, page: int) -> int:
    duration_sec = int(reel.get("video_duration_sec") or 0)
    if page <= 1:
        return 2
    if page <= 2:
        return 3
    if page <= 3:
        return 5
    return 10 if duration_sec > 20 * 60 else 8


def _request_effective_min_relevance(
    min_relevance: float | None,
    *,
    page: int,
    subject_tag: str | None,
    strict_topic_only: bool,
) -> float | None:
    requested = float(min_relevance or 0.0)
    if not strict_topic_only or not subject_tag:
        return float(min_relevance) if min_relevance is not None else None
    if page <= 1:
        return max(requested, 0.3)
    if page <= 2:
        return max(requested, 0.22)
    if page <= 5:
        return max(requested - 0.04, 0.24)
    return max(requested - 0.08, 0.2)


def _shape_request_page_reels(
    ranked: list[dict[str, Any]],
    *,
    page: int,
    limit: int,
    subject_tag: str | None,
    strict_topic_only: bool,
    min_relevance: float | None,
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
) -> list[dict[str, Any]]:
    target_visible_count = max(1, int(page or 1) * max(1, int(limit or 1)))
    clip_min = target_clip_duration_min_sec
    clip_max = target_clip_duration_max_sec

    serialized = _serialize_reels_for_request(
        ranked,
        min_relevance=min_relevance,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=clip_min,
        target_clip_duration_max_sec=clip_max,
    )
    shaped = _shape_reels_for_request_context(
        serialized,
        page=page,
        limit=limit,
        subject_tag=subject_tag,
        strict_topic_only=strict_topic_only,
    )

    widened_clip_min = clip_min
    widened_clip_max = clip_max
    if len(shaped) < target_visible_count and (clip_min is not None or clip_max is not None):
        if page <= 2:
            widened_clip_min = max(0, int(clip_min or 0) - 8) if clip_min is not None else None
            widened_clip_max = int(clip_max or 0) + 12 if clip_max is not None else None
        else:
            widened_clip_min = max(0, int(clip_min or 0) - 15) if clip_min is not None else None
            widened_clip_max = int(clip_max or 0) + 20 if clip_max is not None else None
        if widened_clip_min != clip_min or widened_clip_max != clip_max:
            widened_serialized = _serialize_reels_for_request(
                ranked,
                min_relevance=min_relevance,
                preferred_video_duration=preferred_video_duration,
                target_clip_duration_sec=target_clip_duration_sec,
                target_clip_duration_min_sec=widened_clip_min,
                target_clip_duration_max_sec=widened_clip_max,
            )
            widened_shaped = _shape_reels_for_request_context(
                widened_serialized,
                page=page,
                limit=limit,
                subject_tag=subject_tag,
                strict_topic_only=strict_topic_only,
            )
            shaped = _merge_request_reel_lists(shaped, widened_shaped)

    if len(shaped) < target_visible_count:
        relaxed_serialized = _filter_reels_by_min_relevance(ranked, min_relevance)
        relaxed_shaped = _shape_reels_for_request_context(
            relaxed_serialized,
            page=page,
            limit=limit,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
        )
        shaped = _merge_request_reel_lists(shaped, relaxed_shaped)

    if page <= 1 and strict_topic_only and subject_tag and len(shaped) < max(1, limit):
        emergency_min_relevance = max(0.27, float(min_relevance or 0.3) - 0.03)
        emergency_serialized = _serialize_reels_for_request(
            ranked,
            min_relevance=emergency_min_relevance,
            preferred_video_duration=preferred_video_duration,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        emergency_shaped = _shape_reels_for_request_context(
            emergency_serialized,
            page=1,
            limit=limit,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
        )
        shaped = _merge_request_reel_lists(shaped, emergency_shaped)

    if len(shaped) < target_visible_count:
        fallback_shaped = _shape_reels_for_request_context(
            ranked,
            page=page,
            limit=limit,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
        )
        shaped = _merge_request_reel_lists(shaped, fallback_shaped)

    return shaped


def _shape_reels_for_request_context(
    reels: list[dict[str, Any]],
    *,
    page: int,
    limit: int,
    subject_tag: str | None,
    strict_topic_only: bool,
) -> list[dict[str, Any]]:
    if not reels:
        return []
    if all(bool(reel.get("_selection_ordered")) for reel in reels):
        # The confidence-contract scheduler already applied global priority and
        # prerequisite topology. Request-keyword sorting, similarity filters,
        # and source caps can only corrupt that order or remove a prerequisite.
        return [dict(reel) for reel in reels]

    root_terms, companion_terms, page_one_companion_terms = _request_root_anchor_terms(subject_tag)
    curated_broad_topic = bool(page_one_companion_terms)
    similarity_profile = _request_similarity_profile(subject_tag)
    eligible: list[dict[str, Any]] = []
    for reel in reels:
        if not _request_source_surface_allowed(reel, page=page):
            continue
        if reel_service._is_hard_blocked_low_value_video(
            title=str(reel.get("video_title") or ""),
            description=str(reel.get("video_description") or ""),
            channel_title=str(reel.get("channel_title") or ""),
            subject_tag=subject_tag,
        ):
            continue

        has_root_anchor = False
        has_companion_anchor = False
        has_page_one_companion_anchor = False
        educational_support = False
        topical_short_support = False
        if strict_topic_only and subject_tag:
            text = _request_reel_text(reel)
            video_duration_sec = max(0, int(reel.get("video_duration_sec") or 0))
            relevance_score = float(reel.get("relevance_score") or 0.0)
            matched_terms = reel.get("matched_terms") or []
            if not isinstance(matched_terms, list):
                matched_terms = []
            has_root_anchor = _request_text_has_anchor(text, root_terms)
            has_companion_anchor = _request_text_has_anchor(text, companion_terms)
            has_page_one_companion_anchor = _request_text_has_anchor(text, page_one_companion_terms)
            educational_support = (
                relevance_score >= (0.26 if page <= 1 else 0.22 if page <= 3 else 0.18)
                or bool(
                    re.findall(
                        r"\b("
                        r"lecture|tutorial|explained|introduction|science|study|education|biology|"
                        r"history|behavior|behaviour|societ(?:y|ies)|colony|colonies|world|kingdom|"
                        r"conversation|conference|episode|documentary|macro|fundamentals|basics|overview"
                        r")\b",
                        text,
                    )
                )
            )
            topical_short_support = (
                (0 < video_duration_sec <= 3 * 60)
                and (has_root_anchor or has_companion_anchor or has_page_one_companion_anchor)
                and (
                    relevance_score >= (0.18 if page <= 1 else 0.14 if page <= 3 else 0.12)
                    or len([term for term in matched_terms if str(term or "").strip()]) >= 2
                )
            )
            # The practice selector has already grounded and quality-checked
            # the clip against its transcript. These request-context signals
            # only reorder topic reels; they must not erase valid inventory.

        shaped = dict(reel)
        shaped["_request_rank_score"] = _request_rank_score(
            reel,
            page=page,
            has_root_anchor=has_root_anchor,
            has_companion_anchor=has_companion_anchor,
            has_page_one_companion_anchor=has_page_one_companion_anchor,
            educational_support=educational_support or topical_short_support,
        )
        eligible.append(shaped)

    eligible.sort(key=lambda item: (float(item.get("_request_rank_score") or 0.0), str(item.get("created_at") or "")), reverse=True)

    shaped_rows: list[dict[str, Any]] = []
    per_video_counts: dict[str, int] = {}
    per_anchor_counts: dict[str, int] = {}
    request_window = max(1, int(page or 1) * max(1, int(limit or 1)))
    for item in eligible:
        video_url = str(item.get("video_url") or "")
        match = re.search(r"/embed/([^?&/]+)", video_url)
        video_id = match.group(1) if match else ""
        similarity_threshold = float(similarity_profile.get("cross_video_similarity") or 0.92)
        if page <= 1:
            similarity_threshold = min(0.97, similarity_threshold + 0.03)
        elif page <= 2:
            similarity_threshold = min(0.96, similarity_threshold + 0.02)
        elif page <= 3:
            similarity_threshold = min(0.94, similarity_threshold + 0.01)
        else:
            similarity_threshold = max(0.84, similarity_threshold - 0.02)
        if any(_request_reel_similarity(item, prev) >= similarity_threshold for prev in shaped_rows):
            continue
        if video_id:
            current_count = per_video_counts.get(video_id, 0)
            if current_count >= _request_page_video_cap(item, page=page):
                continue
            if current_count >= (2 if page <= 1 else 3 if page <= 2 else 4 if page <= 3 else 6) and not _request_diversity_ready(
                per_video_counts=per_video_counts,
                per_anchor_counts=per_anchor_counts,
            ):
                continue
            per_video_counts[video_id] = current_count + 1
        if curated_broad_topic and strict_topic_only and subject_tag:
            text = _request_reel_text(item)
            matched_companions = _request_matched_anchor_terms(text, page_one_companion_terms)
            matched_roots = _request_matched_anchor_terms(text, root_terms)
            candidate_anchor_keys = matched_companions or matched_roots or [str(subject_tag).strip()]
            anchor_cap = 4 if page <= 1 else 5 if page <= 2 else 8 if page <= 3 else 12
            chosen_anchor = None
            for candidate_anchor in candidate_anchor_keys:
                if (
                    per_anchor_counts.get(candidate_anchor, 0) >= (3 if page <= 1 else 4 if page <= 2 else 6)
                    and not _request_diversity_ready(
                        per_video_counts=per_video_counts,
                        per_anchor_counts=per_anchor_counts,
                    )
                ):
                    continue
                if per_anchor_counts.get(candidate_anchor, 0) < anchor_cap:
                    chosen_anchor = candidate_anchor
                    break
            if chosen_anchor is None:
                continue
            per_anchor_counts[chosen_anchor] = per_anchor_counts.get(chosen_anchor, 0) + 1
        clean_item = dict(item)
        clean_item.pop("_request_rank_score", None)
        shaped_rows.append(clean_item)
        if len(shaped_rows) >= max(len(eligible), request_window + 12):
            break

    return shaped_rows


def _search_context_has_usable_boundary(
    context: Any,
    *,
    t_start: object | None = None,
    t_end: object | None = None,
) -> bool:
    if not isinstance(context, dict):
        return False
    if _gemini_selection_is_authoritative(context):
        try:
            start = float(t_start)
            end = float(t_end)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(start)
            and math.isfinite(end)
            and start >= 0.0
            and end > start
        )
    if not persisted_boundary_is_usable(context, t_start=t_start, t_end=t_end):
        return False
    return bool(
        str(context.get("selection_contract_version") or "").strip()
        == SELECTION_CONTRACT_VERSION
        and context.get("speech_corridor_verified") is True
    )


def _gemini_selection_is_authoritative(context: object) -> bool:
    return bool(
        isinstance(context, dict)
        and str(context.get("selection_authority") or "").strip().casefold()
        == "gemini"
    )


def _count_generation_reels(
    conn,
    generation_id: str,
) -> int:
    try:
        rows = fetch_all(
            conn,
            "SELECT t_start, t_end, search_context_json FROM reels WHERE generation_id = ?",
            (generation_id,),
        )
    except Exception as exc:
        if "search_context_json" not in str(exc).lower():
            raise
        return 0
    count = 0
    for row in rows:
        try:
            context = json.loads(str(row.get("search_context_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            context = {}
        if not isinstance(context, dict):
            continue
        surface_eligible = context.get("surface_eligible")
        if isinstance(surface_eligible, str):
            surface_eligible = surface_eligible.strip().lower() in {
                "1", "true", "yes", "on",
            }
        surface_reason = str(context.get("surface_reason") or "").strip().lower()
        if (
            not _gemini_selection_is_authoritative(context)
            and surface_eligible is not True
            and surface_reason != "level_mismatch"
        ):
            continue
        if not _search_context_has_usable_boundary(
            context, t_start=row.get("t_start"), t_end=row.get("t_end")
        ):
            continue
        count += 1
    return count


def _count_generation_surfaceable_reels(conn, generation_id: str) -> int:
    """Count usable clips; difficulty affects order, never eligibility."""

    return _count_generation_reels(conn, generation_id)


def _count_material_ready_reels(conn, material_id: str) -> int:
    rows = fetch_all(
        conn,
        "SELECT t_start, t_end, search_context_json FROM reels WHERE material_id = ?",
        (material_id,),
    )
    count = 0
    for row in rows:
        try:
            context = json.loads(str(row.get("search_context_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        surface_eligible = context.get("surface_eligible") if isinstance(context, dict) else None
        if isinstance(surface_eligible, str):
            surface_eligible = surface_eligible.strip().lower() in {
                "1", "true", "yes", "on",
            }
        surface_reason = (
            str(context.get("surface_reason") or "").strip().lower()
            if isinstance(context, dict)
            else ""
        )
        if (
            _gemini_selection_is_authoritative(context)
            or surface_eligible is True
            or surface_reason == "level_mismatch"
        ) and _search_context_has_usable_boundary(
            context, t_start=row.get("t_start"), t_end=row.get("t_end")
        ):
            count += 1
    return count


def _usable_boundary_reel_ids(conn, reel_ids: list[str]) -> set[str]:
    normalized = list(dict.fromkeys(
        str(reel_id or "").strip() for reel_id in reel_ids if str(reel_id or "").strip()
    ))
    if not normalized:
        return set()
    placeholders = ", ".join("?" for _ in normalized)
    rows = fetch_all(
        conn,
        f"SELECT id, t_start, t_end, search_context_json FROM reels WHERE id IN ({placeholders})",
        tuple(normalized),
    )
    usable: set[str] = set()
    for row in rows:
        try:
            context = json.loads(str(row.get("search_context_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        surface_eligible = (
            context.get("surface_eligible") if isinstance(context, dict) else False
        )
        if isinstance(surface_eligible, str):
            surface_eligible = surface_eligible.strip().lower() in {
                "1", "true", "yes", "on",
            }
        surface_reason = (
            str(context.get("surface_reason") or "").strip().lower()
            if isinstance(context, dict)
            else ""
        )
        if (
            _gemini_selection_is_authoritative(context)
            or surface_eligible is True
            or surface_reason == "level_mismatch"
        ) and (
            _search_context_has_usable_boundary(
                context, t_start=row.get("t_start"), t_end=row.get("t_end")
            )
        ):
            usable.add(str(row.get("id") or ""))
    return usable


def _fetch_generation_row(conn, generation_id: str | None) -> dict[str, Any] | None:
    if not generation_id:
        return None
    return fetch_one(conn, "SELECT * FROM reel_generations WHERE id = ?", (generation_id,))


def _stored_generation_lesson_order_metadata(
    conn,
    generation_id: str | None,
) -> dict[str, Any] | None:
    """Return persisted metadata, None for legacy, or {} for corruption."""
    row = _fetch_generation_row(conn, generation_id)
    return _parse_generation_lesson_order_metadata(
        (row or {}).get("lesson_order_json"),
        generation_id=generation_id,
    )


def _parse_generation_lesson_order_metadata(
    raw: object,
    *,
    generation_id: str | None,
) -> dict[str, Any] | None:
    """Parse already-loaded lesson metadata with legacy/corruption semantics."""
    if raw is None:
        return None
    try:
        payload = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        logger.warning("Rejecting malformed lesson order generation_id=%s", generation_id)
        return {}
    if not isinstance(payload, dict):
        logger.warning("Rejecting malformed lesson order generation_id=%s", generation_id)
        return {}
    return payload


def _stored_generation_lesson_order_ids(
    conn,
    generation_id: str | None,
) -> list[str] | None:
    """Return a release order, None for legacy, or [] for organizer-only/invalid."""
    payload = _stored_generation_lesson_order_metadata(conn, generation_id)
    if payload is None:
        return None
    values = payload.get("ordered_reel_ids")
    if not isinstance(values, list):
        return []
    ordered_ids = [value if isinstance(value, str) else "" for value in values]
    if not ordered_ids and _lesson_reconciliation_tail_ids(payload):
        return []
    if (
        not ordered_ids
        or any(not reel_id for reel_id in ordered_ids)
        or len(set(ordered_ids)) != len(ordered_ids)
    ):
        logger.warning("Rejecting invalid lesson order generation_id=%s", generation_id)
        return []
    return ordered_ids


def _lesson_reconciliation_tail_ids(
    metadata: dict[str, Any] | None,
) -> list[str] | None:
    """Return a validated optional cross-batch unseen-tail order."""
    if not isinstance(metadata, dict):
        return None
    values = metadata.get("reconciliation_tail_reel_ids")
    if values is None:
        return None
    if not isinstance(values, list):
        return []
    tail_ids = [value if isinstance(value, str) else "" for value in values]
    current_ids = metadata.get("ordered_reel_ids")
    if (
        not tail_ids
        or any(not reel_id.strip() for reel_id in tail_ids)
        or len(set(tail_ids)) != len(tail_ids)
        or not isinstance(current_ids, list)
        or any(
            not isinstance(reel_id, str) or not reel_id.strip()
            for reel_id in current_ids
        )
        or len(set(current_ids)) != len(current_ids)
        or not set(current_ids).issubset(tail_ids)
    ):
        return []
    return tail_ids


def _stored_generation_reconciliation_tail_ids(
    conn,
    generation_id: str | None,
) -> list[str] | None:
    return _lesson_reconciliation_tail_ids(
        _stored_generation_lesson_order_metadata(conn, generation_id)
    )


def _apply_reconciliation_tail_order(
    existing_ids: list[str],
    metadata: dict[str, Any] | None,
) -> list[str]:
    """Overlay a validated unseen-tail plan without moving unmentioned history."""
    tail_ids = _lesson_reconciliation_tail_ids(metadata)
    if not tail_ids:
        return existing_ids
    existing_id_set = set(existing_ids)
    if not set(tail_ids).issubset(existing_id_set):
        return existing_ids
    tail_id_set = set(tail_ids)
    return [
        reel_id for reel_id in existing_ids if reel_id not in tail_id_set
    ] + tail_ids


def _surviving_terminal_summary_start_reel_id(
    *,
    ordered_reel_ids: Iterable[str],
    terminal_summary_start_reel_id: object,
    surviving_reel_ids: Iterable[str],
) -> str | None:
    """Project an organizer recap suffix marker onto the surviving release."""
    marker = (
        terminal_summary_start_reel_id.strip()
        if isinstance(terminal_summary_start_reel_id, str)
        else ""
    )
    ordered_ids = [
        reel_id.strip()
        for reel_id in ordered_reel_ids
        if isinstance(reel_id, str) and reel_id.strip()
    ]
    if not marker or marker not in ordered_ids:
        return None
    surviving_ids = {
        reel_id.strip()
        for reel_id in surviving_reel_ids
        if isinstance(reel_id, str) and reel_id.strip()
    }
    marker_index = ordered_ids.index(marker)
    return next(
        (
            reel_id
            for reel_id in ordered_ids[marker_index:]
            if reel_id in surviving_ids
        ),
        None,
    )


def _apply_generation_lesson_order(
    conn,
    *,
    generation_id: str | None,
    reels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply or project the persisted selection onto the current valid reels."""
    ordered_ids = _stored_generation_lesson_order_ids(conn, generation_id)
    if ordered_ids is None:
        return reels
    if not reels:
        return []
    reel_ids = [
        value if isinstance(value, str) else ""
        for reel in reels
        for value in [reel.get("reel_id")]
    ]
    if (
        not ordered_ids
        or any(not reel_id for reel_id in reel_ids)
        or len(set(reel_ids)) != len(reel_ids)
    ):
        logger.warning(
            "Ignoring invalid lesson selection generation_id=%s stored=%d valid=%d",
            generation_id,
            len(ordered_ids),
            len(reel_ids),
        )
        return reels
    by_id = dict(zip(reel_ids, reels, strict=True))
    stored_id_set = set(ordered_ids)
    current_id_set = set(reel_ids)
    if stored_id_set.issubset(current_id_set):
        return [by_id[reel_id] for reel_id in ordered_ids]
    if current_id_set.issubset(stored_id_set):
        return [by_id[reel_id] for reel_id in ordered_ids if reel_id in by_id]
    logger.warning(
        "Ignoring invalid lesson selection generation_id=%s stored=%d valid=%d",
        generation_id,
        len(ordered_ids),
        len(reel_ids),
    )
    return reels


def _persist_generation_lesson_order(
    conn,
    *,
    generation_id: str,
    metadata: dict[str, Any],
) -> None:
    updated = execute_modify(
        conn,
        "UPDATE reel_generations SET lesson_order_json = ? WHERE id = ?",
        (dumps_json(metadata), generation_id),
    )
    if not updated:
        raise RuntimeError(
            f"generation row not found while storing lesson order: {generation_id}"
        )


def _lesson_order_topic(conn, *, material_id: str, reels: list[dict[str, Any]]) -> str:
    material = fetch_one(
        conn,
        "SELECT raw_text, subject_tag, source_type FROM materials WHERE id = ?",
        (material_id,),
    ) or {}
    if str(material.get("source_type") or "").strip().casefold() == "topic":
        topic = " ".join(str(material.get("raw_text") or "").split())
        if topic:
            return topic[:500]
    titles = list(
        dict.fromkeys(
            title
            for reel in reels
            if (title := str(reel.get("concept_title") or "").strip())
        )
    )
    return " / ".join(titles[:6]) or str(material.get("subject_tag") or material_id)


def _fetch_generation_head_row(conn, *, material_id: str, request_key: str) -> dict[str, Any] | None:
    return fetch_one(
        conn,
        "SELECT * FROM reel_generation_heads WHERE material_id = ? AND request_key = ?",
        (material_id, request_key),
    )


def _fetch_active_generation_row(conn, *, material_id: str, request_key: str) -> dict[str, Any] | None:
    head = _fetch_generation_head_row(conn, material_id=material_id, request_key=request_key)
    if not head:
        return None
    return _fetch_generation_row(conn, str(head.get("active_generation_id") or ""))


def _create_generation_row(
    conn,
    *,
    material_id: str,
    concept_id: str | None,
    request_key: str,
    generation_mode: Literal["slow", "fast"],
    retrieval_profile: str,
    source_generation_id: str | None = None,
    generation_id: str | None = None,
) -> str:
    resolved_generation_id = str(generation_id or uuid.uuid4())
    upsert(
        conn,
        "reel_generations",
        {
            "id": resolved_generation_id,
            "material_id": material_id,
            "concept_id": concept_id,
            "request_key": request_key,
            "generation_mode": generation_mode,
            "retrieval_profile": retrieval_profile,
            "status": "pending",
            "source_generation_id": source_generation_id,
            "reel_count": 0,
            "created_at": now_iso(),
            "completed_at": None,
            "activated_at": None,
            "error_text": None,
        },
    )
    return resolved_generation_id


def _complete_generation(
    conn,
    *,
    generation_id: str,
    retrieval_profile: str,
    status: str,
    error_text: str | None = None,
    activate: bool = False,
) -> None:
    row = _fetch_generation_row(conn, generation_id) or {}
    completed_at = now_iso()
    activated_at = completed_at if activate else row.get("activated_at")
    upsert(
        conn,
        "reel_generations",
        {
            "id": generation_id,
            "material_id": str(row.get("material_id") or ""),
            "concept_id": row.get("concept_id"),
            "request_key": str(row.get("request_key") or ""),
            "generation_mode": str(row.get("generation_mode") or "slow"),
            "retrieval_profile": retrieval_profile,
            "status": status,
            "source_generation_id": row.get("source_generation_id"),
            "reel_count": _count_generation_reels(conn, generation_id),
            "created_at": str(row.get("created_at") or completed_at),
            "completed_at": completed_at,
            "activated_at": activated_at,
            "error_text": error_text,
        },
    )


def _activate_generation(conn, *, material_id: str, request_key: str, generation_id: str, retrieval_profile: str) -> None:
    _complete_generation(
        conn,
        generation_id=generation_id,
        retrieval_profile=retrieval_profile,
        status="active",
        activate=True,
    )
    updated_at = now_iso()
    try:
        upsert(
            conn,
            "reel_generation_heads",
            {
                "id": _build_generation_head_id(material_id, request_key),
                "material_id": material_id,
                "request_key": request_key,
                "active_generation_id": generation_id,
                "updated_at": updated_at,
            },
        )
    except sqlite3.OperationalError as exc:
        # Older local SQLite databases used a composite primary key and never had
        # the surrogate `id` column. Keep that schema working in-place.
        if "no column named id" not in str(exc).lower():
            raise
        execute_modify(
            conn,
            """
            INSERT INTO reel_generation_heads (material_id, request_key, active_generation_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(material_id, request_key) DO UPDATE SET
                active_generation_id = excluded.active_generation_id,
                updated_at = excluded.updated_at
            """,
            (material_id, request_key, generation_id, updated_at),
        )
def _normalize_reel_identity_time(value: object) -> str:
    try:
        parsed = round(float(value or 0.0), 3)
    except (TypeError, ValueError):
        parsed = 0.0
    return f"{parsed:.3f}"


def _reel_source_video_id(reel: dict[str, Any]) -> str:
    video_url = str(reel.get("video_url") or "").strip()
    video_identity = str(reel.get("video_id") or "").strip()
    if not video_identity:
        embed_match = re.search(r"/embed/([^?&/]+)", video_url)
        if embed_match:
            video_identity = embed_match.group(1)
        else:
            parsed = urlparse(video_url)
            query_video_id = parse_qs(parsed.query).get("v", [""])[0]
            video_identity = query_video_id or parsed.path or video_url
    return video_identity


_LESSON_ORDER_SELECTION_FIELDS = frozenset({
    "_selection_candidate_id",
    "_selection_chain_id",
    "_selection_chain_position",
    "_selection_prerequisite_ids",
    "_selection_topic_relevance",
    "_selection_informativeness",
    "_selection_concept",
    "_selection_concept_family",
    "_selection_concept_aliases",
    "_selection_intent_obligations",
    "_selection_intent_connections",
    "_selection_intent_relationship_witnesses",
    "_selection_intent_curriculum_edges",
    "_selection_intent_role",
    "_selection_intent_coverage",
    "_selection_directly_teaches_topic",
    "_selection_boundary_status",
    "_selection_self_contained",
    "_selection_is_standalone",
})


def _public_generation_reel(
    reel: dict[str, Any],
    *,
    preserve_lesson_order_metadata: bool = False,
) -> dict[str, Any]:
    public_reel = dict(reel)
    selection_contract_version = str(
        public_reel.get("selection_contract_version")
        or public_reel.get("_selection_contract_version")
        or ""
    ).strip()
    if selection_contract_version:
        public_reel["selection_contract_version"] = selection_contract_version
    selector_relevance = public_reel.get("_selection_topic_relevance")
    if selector_relevance is None:
        selector_relevance = public_reel.get("topic_relevance")
    if selector_relevance is not None:
        try:
            parsed_selector_relevance = float(selector_relevance)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if math.isfinite(parsed_selector_relevance):
                parsed_selector_relevance = max(
                    0.0, min(1.0, parsed_selector_relevance)
                )
                public_reel["topic_relevance"] = parsed_selector_relevance
                if selection_contract_version == SELECTION_CONTRACT_VERSION:
                    public_reel["relevance_score"] = parsed_selector_relevance
    public_reel["video_id"] = _reel_source_video_id(public_reel)
    return {
        key: value
        for key, value in public_reel.items()
        if not key.startswith("_selection_")
        or (
            preserve_lesson_order_metadata
            and key in _LESSON_ORDER_SELECTION_FIELDS
        )
    }


def _current_selection_contract_reels(
    reels: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fail closed when stale inventory reaches a current request path."""
    return [
        reel
        for reel in reels
        if str(
            reel.get("_selection_contract_version")
            or reel.get("selection_contract_version")
            or ""
        ).strip()
        == SELECTION_CONTRACT_VERSION
    ]


def _prefer_proven_boundary_inventory(
    reels: Iterable[dict[str, Any]],
    *,
    protected_reel_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Use coarse Gemini cuts only when no stronger clip is available.

    ``best_effort`` rows remain valid fallback inventory.  A previously
    released row is immutable, but a fresh best-effort row must not displace a
    verified or transcript-context-aligned candidate merely to fill a batch.
    """

    candidates = list(reels)
    protected = {
        reel_id
        for value in protected_reel_ids
        if (reel_id := str(value or "").strip())
    }
    stronger_boundary_statuses = {"verified", "context_aligned"}
    has_stronger_candidate = any(
        str(reel.get("_selection_boundary_status") or "")
        .strip()
        .casefold()
        in stronger_boundary_statuses
        for reel in candidates
    )
    if not has_stronger_candidate:
        return candidates
    return [
        reel
        for reel in candidates
        if str(reel.get("reel_id") or "").strip() in protected
        or str(reel.get("_selection_boundary_status") or "")
        .strip()
        .casefold()
        != "best_effort"
    ]


def _reel_identity_key(reel: dict[str, Any]) -> tuple[str, str]:
    reel_id = str(reel.get("reel_id") or "").strip()
    video_identity = _reel_source_video_id(reel)
    clip_key = ":".join(
        [
            video_identity,
            _normalize_reel_identity_time(reel.get("t_start")),
            _normalize_reel_identity_time(reel.get("t_end")),
        ]
    )
    return reel_id, clip_key


def _merge_request_reel_lists(*reel_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_reel_ids: set[str] = set()
    seen_clip_keys: set[str] = set()
    for reel_list in reel_lists:
        for reel in reel_list:
            reel_id, clip_key = _reel_identity_key(reel)
            if reel_id and reel_id in seen_reel_ids:
                continue
            if clip_key in seen_clip_keys:
                continue
            if reel_id:
                seen_reel_ids.add(reel_id)
            seen_clip_keys.add(clip_key)
            merged.append(reel)

    return merged


def _merge_selection_ordered_reel_lists(
    *reel_lists: list[dict[str, Any]],
    target_stage: int | None = None,
) -> list[dict[str, Any]]:
    """Quality-merge topologically ordered generation batches.

    Only each batch head is eligible, so a dependent can never jump ahead of
    a prerequisite that the service placed before it.
    """
    positions = [0 for _reel_list in reel_lists]
    input_offsets: list[int] = []
    offset = 0
    for reel_list in reel_lists:
        input_offsets.append(offset)
        offset += len(reel_list)

    merged: list[dict[str, Any]] = []
    seen_reel_ids: set[str] = set()
    seen_clip_keys: set[str] = set()
    while True:
        heads = [
            (
                reel_service._selection_contract_sort_key(
                    reel_list[positions[batch_index]],
                    input_order=input_offsets[batch_index] + positions[batch_index],
                    target_stage=target_stage,
                ),
                batch_index,
            )
            for batch_index, reel_list in enumerate(reel_lists)
            if positions[batch_index] < len(reel_list)
        ]
        if not heads:
            return merged
        _priority, batch_index = min(heads)
        reel = reel_lists[batch_index][positions[batch_index]]
        positions[batch_index] += 1
        reel_id, clip_key = _reel_identity_key(reel)
        if (reel_id and reel_id in seen_reel_ids) or clip_key in seen_clip_keys:
            continue
        if reel_id:
            seen_reel_ids.add(reel_id)
        seen_clip_keys.add(clip_key)
        merged.append(reel)


def _response_generation_ids(
    conn,
    generation_id: str | None,
    *,
    generation_rows_out: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    if not generation_id:
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    def collect(current_generation_id: str) -> None:
        if not current_generation_id or current_generation_id in seen:
            return
        seen.add(current_generation_id)
        generation_row = _fetch_generation_row(conn, current_generation_id)
        if not generation_row:
            ordered.append(current_generation_id)
            return
        if generation_rows_out is not None:
            generation_rows_out[current_generation_id] = generation_row
        source_generation_id = str(generation_row.get("source_generation_id") or "").strip()
        if source_generation_id:
            collect(source_generation_id)
        ordered.append(str(generation_row.get("id") or current_generation_id))

    collect(generation_id)
    return ordered


def _same_adaptation_current_restatement_policy(
    generation_rows: Mapping[str, Mapping[str, Any]],
    generation_ids: Iterable[str],
    *,
    adaptation_fingerprint: object,
    reacquisition_checkpoint_out: list[bool] | None = None,
) -> tuple[set[str], set[str]]:
    """Return exact organizer omissions and their selected availability guards."""
    fingerprint = str(adaptation_fingerprint or "").strip()
    if not fingerprint:
        if reacquisition_checkpoint_out is not None:
            reacquisition_checkpoint_out.append(False)
        return set(), set()
    restatement_ids: set[str] = set()
    guard_ids: set[str] = set()
    reacquisition_checkpoint_seen = False
    for generation_id in generation_ids:
        row = generation_rows.get(str(generation_id or "").strip())
        if not isinstance(row, Mapping):
            continue
        metadata = _parse_generation_lesson_order_metadata(
            row.get("lesson_order_json"),
            generation_id=str(generation_id or "").strip(),
        )
        if (
            not isinstance(metadata, dict)
            or str(metadata.get("adaptation_fingerprint") or "").strip()
            != fingerprint
        ):
            continue
        raw_ids = metadata.get("current_restatement_reel_ids")
        raw_guards = metadata.get("current_restatement_guard_reel_ids")
        if not isinstance(raw_ids, list) or not isinstance(raw_guards, list):
            continue
        normalized = [
            value.strip() if isinstance(value, str) else ""
            for value in raw_ids
        ]
        normalized_guards = [
            value.strip() if isinstance(value, str) else ""
            for value in raw_guards
        ]
        if (
            any(not reel_id for reel_id in normalized)
            or len(set(normalized)) != len(normalized)
            or not normalized_guards
            or any(not reel_id for reel_id in normalized_guards)
            or len(set(normalized_guards)) != len(normalized_guards)
            or not set(normalized).isdisjoint(normalized_guards)
        ):
            logger.warning(
                "Ignoring invalid current restatements generation_id=%s",
                generation_id,
            )
            continue
        restatement_ids.update(normalized)
        guard_ids.update(normalized_guards)
        reacquisition_checkpoint_seen = bool(
            reacquisition_checkpoint_seen
            or metadata.get("editorial_reacquisition_checkpoint") is True
        )
    if reacquisition_checkpoint_out is not None:
        reacquisition_checkpoint_out.append(reacquisition_checkpoint_seen)
    if not restatement_ids or not guard_ids:
        return set(), set()
    return restatement_ids, guard_ids


def _generation_chain_rows_snapshot(
    conn,
    generation_ids: Iterable[str | None],
) -> dict[str, dict[str, Any]]:
    """Load multiple generation ancestry chains in one cycle-safe query."""
    anchors = list(dict.fromkeys(
        str(generation_id or "").strip()
        for generation_id in generation_ids
        if str(generation_id or "").strip()
    ))
    if not anchors:
        return {}
    placeholders = ", ".join("?" for _generation_id in anchors)
    rows = fetch_all(
        conn,
        f"""
        WITH RECURSIVE generation_chain(
            id, source_generation_id, lesson_order_json
        ) AS (
            SELECT id, source_generation_id, lesson_order_json
            FROM reel_generations
            WHERE id IN ({placeholders})
            UNION
            SELECT parent.id, parent.source_generation_id,
                   parent.lesson_order_json
            FROM reel_generations AS parent
            JOIN generation_chain AS child
              ON parent.id = child.source_generation_id
        )
        SELECT id, source_generation_id, lesson_order_json
        FROM generation_chain
        """,
        tuple(anchors),
    )
    return {
        str(row.get("id") or "").strip(): row
        for row in rows
        if str(row.get("id") or "").strip()
    }


def _snapshot_generation_ids(
    rows_by_id: dict[str, dict[str, Any]],
    generation_id: str | None,
) -> list[str]:
    """Rebuild the existing oldest-to-newest ancestry semantics in memory."""
    current = str(generation_id or "").strip()
    if not current:
        return []
    newest_to_oldest: list[str] = []
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        newest_to_oldest.append(current)
        row = rows_by_id.get(current)
        if row is None:
            break
        current = str(row.get("source_generation_id") or "").strip()
    return list(reversed(newest_to_oldest))


def _authoritative_generation_releases_snapshot(
    conn,
    generation_ids: list[str],
) -> dict[str, list[str]]:
    """Load each generation's latest authoritative terminal release at once."""
    if not generation_ids:
        return {}
    placeholders = ", ".join("?" for _generation_id in generation_ids)
    rows = fetch_all(
        conn,
        f"""
        WITH ranked_jobs AS (
            SELECT id, result_generation_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY result_generation_id
                       ORDER BY completed_at DESC, updated_at DESC,
                                created_at DESC, id DESC
                   ) AS job_rank
            FROM reel_generation_jobs
            WHERE result_generation_id IN ({placeholders})
              AND status IN ('completed', 'partial', 'exhausted')
        ), ranked_events AS (
            SELECT jobs.result_generation_id, jobs.id AS job_id,
                   jobs.job_rank, events.seq, events.event_type,
                   events.payload_json,
                   ROW_NUMBER() OVER (
                       PARTITION BY events.job_id ORDER BY events.seq
                   ) AS event_rank
            FROM ranked_jobs AS jobs
            JOIN generation_job_events AS events ON events.job_id = jobs.id
            WHERE jobs.job_rank <= 20 AND events.seq > 0
        )
        SELECT result_generation_id, job_id, job_rank, seq,
               event_type, payload_json
        FROM ranked_events
        WHERE event_rank <= 500
        ORDER BY result_generation_id, job_rank, seq
        """,
        tuple(generation_ids),
    )
    events_by_job: dict[str, list[dict[str, Any]]] = {}
    job_order_by_generation: dict[str, list[str]] = {}
    for row in rows:
        generation_id = str(row.get("result_generation_id") or "").strip()
        job_id = str(row.get("job_id") or "").strip()
        if not generation_id or not job_id:
            continue
        job_order = job_order_by_generation.setdefault(generation_id, [])
        if job_id not in events_by_job:
            events_by_job[job_id] = []
            job_order.append(job_id)
        events_by_job[job_id].append(row)

    releases: dict[str, list[str]] = {}
    for generation_id in generation_ids:
        for job_id in job_order_by_generation.get(generation_id, ()):
            authoritative_payload: dict[str, Any] | None = None
            for event in reversed(events_by_job.get(job_id, ())):
                if str(event.get("event_type") or "") != "final":
                    continue
                raw_payload = event.get("payload_json")
                try:
                    payload = (
                        raw_payload
                        if isinstance(raw_payload, dict)
                        else json.loads(str(raw_payload or "{}"))
                    )
                except (TypeError, json.JSONDecodeError):
                    payload = {}
                if isinstance(payload, dict) and payload.get("authoritative") is True:
                    authoritative_payload = payload
                    break
            if authoritative_payload is None:
                continue
            ordered_ids: list[str] = []
            seen_ids: set[str] = set()
            raw_reels = authoritative_payload.get("reels")
            if isinstance(raw_reels, list):
                for reel in raw_reels:
                    if not isinstance(reel, dict):
                        continue
                    reel_id = str(reel.get("reel_id") or "").strip()
                    if not reel_id or reel_id in seen_ids:
                        continue
                    seen_ids.add(reel_id)
                    ordered_ids.append(reel_id)
            releases[generation_id] = ordered_ids
            break
    return releases


def _authoritative_release_ids_from_snapshot(
    *,
    generation_ids: list[str],
    generation_rows: dict[str, dict[str, Any]],
    releases_by_generation: dict[str, list[str]],
) -> list[str] | None:
    """Compose teaching then recap suffixes from already-loaded chain state."""
    teaching_ids: list[str] = []
    terminal_summary_ids: list[str] = []
    found_authoritative_release = False
    for generation_id in generation_ids:
        if generation_id not in releases_by_generation:
            continue
        found_authoritative_release = True
        released_ids = releases_by_generation[generation_id]
        lesson_order_metadata = (
            _parse_generation_lesson_order_metadata(
                (generation_rows.get(generation_id) or {}).get(
                    "lesson_order_json"
                ),
                generation_id=generation_id,
            )
            or {}
        )
        stored_order = lesson_order_metadata.get("ordered_reel_ids")
        marker_order = stored_order if isinstance(stored_order, list) else released_ids
        terminal_summary_start_reel_id = (
            _surviving_terminal_summary_start_reel_id(
                ordered_reel_ids=marker_order,
                terminal_summary_start_reel_id=lesson_order_metadata.get(
                    "terminal_summary_start_reel_id"
                ),
                surviving_reel_ids=released_ids,
            )
        )
        if terminal_summary_start_reel_id is None:
            teaching_ids.extend(released_ids)
            continue
        marker_index = released_ids.index(terminal_summary_start_reel_id)
        teaching_ids.extend(released_ids[:marker_index])
        terminal_summary_ids.extend(released_ids[marker_index:])
    if not found_authoritative_release:
        return None
    ordered_ids = list(dict.fromkeys([*teaching_ids, *terminal_summary_ids]))
    for generation_id in generation_ids:
        ordered_ids = _apply_reconciliation_tail_order(
            ordered_ids,
            _parse_generation_lesson_order_metadata(
                (generation_rows.get(generation_id) or {}).get(
                    "lesson_order_json"
                ),
                generation_id=generation_id,
            ),
        )
    return ordered_ids


def _authoritative_generation_release_reel_ids(
    conn,
    generation_id: str,
) -> list[str] | None:
    """Return one generation's exact authoritative release, including empty."""
    jobs = fetch_all(
        conn,
        """
        SELECT id
        FROM reel_generation_jobs
        WHERE result_generation_id = ?
          AND status IN ('completed', 'partial', 'exhausted')
        ORDER BY completed_at DESC, updated_at DESC, created_at DESC, id DESC
        LIMIT 20
        """,
        (generation_id,),
    )
    for job in jobs:
        released_reel_ids = _authoritative_job_release_reel_ids(
            conn,
            str(job.get("id") or ""),
        )
        if released_reel_ids is None:
            continue
        return released_reel_ids
    return None


def _authoritative_job_release_reel_ids(
    conn,
    job_id: str,
) -> list[str] | None:
    """Return one job's authoritative release, including an explicit empty release."""
    authoritative_payload: dict[str, Any] | None = None
    for event in reversed(replay_generation_events(conn, job_id=job_id)):
        if str(event.get("type") or "") != "final":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and payload.get("authoritative") is True:
            authoritative_payload = payload
            break
    if authoritative_payload is None:
        return None
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    reels = authoritative_payload.get("reels")
    if isinstance(reels, list):
        for reel in reels:
            if not isinstance(reel, dict):
                continue
            reel_id = str(reel.get("reel_id") or "").strip()
            if not reel_id or reel_id in seen_ids:
                continue
            seen_ids.add(reel_id)
            ordered_ids.append(reel_id)
    return ordered_ids


def _remove_authoritative_release_temporal_overlaps(
    conn,
    *,
    generation_ids: list[str],
    ordered_reel_ids: list[str],
) -> list[str]:
    """Keep the earliest release when same-source spans substantially overlap."""
    if len(ordered_reel_ids) < 2 or not generation_ids:
        return ordered_reel_ids

    placeholders = ", ".join("?" for _generation_id in generation_ids)
    rows = fetch_all(
        conn,
        f"SELECT id, video_id, video_url, t_start, t_end, search_context_json FROM reels "
        f"WHERE generation_id IN ({placeholders})",
        tuple(generation_ids),
    )
    rows_by_id = {
        str(row.get("id") or "").strip(): row
        for row in rows
        if str(row.get("id") or "").strip()
    }
    reels_by_id: dict[str, dict[str, Any]] = {}
    for reel_id in ordered_reel_ids:
        row = rows_by_id.get(reel_id)
        if row is None:
            reels_by_id[reel_id] = {"reel_id": reel_id}
            continue
        metadata = reel_service._selection_metadata(
            row.get("search_context_json"),
            t_start=row.get("t_start"),
            t_end=row.get("t_end"),
        )
        reels_by_id[reel_id] = {
            **row,
            **metadata,
            "video_id": _bare_video_id(_reel_source_video_id(row)),
        }
    filtered_ids, _checkpoint_ids = _filter_same_source_overlaps(
        ordered_reel_ids,
        (),
        reels_by_id,
    )
    return filtered_ids


def _filter_continuation_release_temporal_overlaps(
    conn,
    *,
    source_generation_id: str | None,
    generation_id: str,
    reels: list[dict[str, Any]],
    prior_reel_ids_out: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Remove child clips already covered by an authoritative source release."""
    if not source_generation_id:
        return reels
    generation_rows = _generation_chain_rows_snapshot(
        conn,
        (source_generation_id, generation_id),
    )
    source_generation_ids = _snapshot_generation_ids(
        generation_rows,
        source_generation_id,
    )
    current_generation_ids = _snapshot_generation_ids(
        generation_rows,
        generation_id,
    )
    releases_by_generation = _authoritative_generation_releases_snapshot(
        conn,
        source_generation_ids,
    )
    prior_reel_ids = _authoritative_release_ids_from_snapshot(
        generation_ids=source_generation_ids,
        generation_rows=generation_rows,
        releases_by_generation=releases_by_generation,
    )
    if not prior_reel_ids:
        return reels
    if prior_reel_ids_out is not None:
        prior_reel_ids_out.extend(
            reel_id
            for reel_id in prior_reel_ids
            if reel_id not in prior_reel_ids_out
        )
    if not reels:
        return reels
    prior_reel_id_set = set(prior_reel_ids)
    current_by_id = {
        reel_id: reel
        for reel in reels
        if (reel_id := str(reel.get("reel_id") or "").strip())
    }
    if not current_by_id:
        return reels
    current_by_id = {
        reel_id: reel
        for reel_id, reel in current_by_id.items()
        if reel_id not in prior_reel_id_set
    }
    if not current_by_id:
        return []
    current_ids = list(current_by_id)
    ordered_ids = list(dict.fromkeys([*prior_reel_ids, *current_ids]))
    kept_ids = set(_remove_authoritative_release_temporal_overlaps(
        conn,
        generation_ids=list(dict.fromkeys([
            *source_generation_ids,
            *current_generation_ids,
        ])),
        ordered_reel_ids=ordered_ids,
    ))
    return [
        current_by_id[reel_id]
        for reel_id in current_ids
        if reel_id in kept_ids
    ]


def _authoritative_release_reel_ids(
    conn,
    generation_id: str | None,
) -> list[str] | None:
    """Return released teaching before recap suffixes across the whole chain."""
    generation_ids = _response_generation_ids(conn, generation_id)
    teaching_ids: list[str] = []
    terminal_summary_ids: list[str] = []
    found_authoritative_release = False
    for current_generation_id in generation_ids:
        released_ids = _authoritative_generation_release_reel_ids(
            conn,
            current_generation_id,
        )
        if released_ids is None:
            continue
        found_authoritative_release = True
        lesson_order_metadata = (
            _stored_generation_lesson_order_metadata(conn, current_generation_id)
            or {}
        )
        stored_order = lesson_order_metadata.get("ordered_reel_ids")
        marker_order = (
            stored_order
            if isinstance(stored_order, list)
            else released_ids
        )
        terminal_summary_start_reel_id = (
            _surviving_terminal_summary_start_reel_id(
                ordered_reel_ids=marker_order,
                terminal_summary_start_reel_id=lesson_order_metadata.get(
                    "terminal_summary_start_reel_id"
                ),
                surviving_reel_ids=released_ids,
            )
        )
        if terminal_summary_start_reel_id is None:
            teaching_ids.extend(released_ids)
            continue
        marker_index = released_ids.index(terminal_summary_start_reel_id)
        teaching_ids.extend(released_ids[:marker_index])
        terminal_summary_ids.extend(released_ids[marker_index:])
    if not found_authoritative_release:
        return None
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for reel_id in [*teaching_ids, *terminal_summary_ids]:
        if reel_id in seen_ids:
            continue
        seen_ids.add(reel_id)
        ordered_ids.append(reel_id)
    for current_generation_id in generation_ids:
        ordered_ids = _apply_reconciliation_tail_order(
            ordered_ids,
            _stored_generation_lesson_order_metadata(
                conn, current_generation_id
            ),
        )
    return _remove_authoritative_release_temporal_overlaps(
        conn,
        generation_ids=generation_ids,
        ordered_reel_ids=ordered_ids,
    )


def _verified_reusable_generation_chain(
    conn,
    *,
    generation_id: str,
    material_id: str,
) -> bool:
    """Return whether a chain contains validated inventory for the current contract."""
    generation_ids = _response_generation_ids(conn, generation_id)
    if not generation_ids:
        return False

    placeholders = ", ".join("?" for _ in generation_ids)
    generation_rows = fetch_all(
        conn,
        f"SELECT id, material_id FROM reel_generations WHERE id IN ({placeholders})",
        tuple(generation_ids),
    )
    if (
        {str(row.get("id") or "") for row in generation_rows} != set(generation_ids)
        or any(str(row.get("material_id") or "") != material_id for row in generation_rows)
    ):
        return False

    reel_rows = fetch_all(
        conn,
        f"SELECT t_start, t_end, search_context_json, transcript_snippet "
        f"FROM reels WHERE generation_id IN ({placeholders})",
        tuple(generation_ids),
    )
    verified_reusable_count = 0
    for row in reel_rows:
        try:
            context = json.loads(str(row.get("search_context_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(context, dict):
            continue
        if (
            str(context.get("selection_contract_version") or "").strip()
            != SELECTION_CONTRACT_VERSION
        ):
            continue
        if _gemini_selection_is_authoritative(context):
            if _search_context_has_usable_boundary(
                context, t_start=row.get("t_start"), t_end=row.get("t_end")
            ):
                verified_reusable_count += 1
            continue
        try:
            quality_scores = (
                float(context.get("informativeness")),
                float(context.get("topic_relevance")),
                float(context.get("educational_importance")),
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            any(not math.isfinite(score) for score in quality_scores)
            or any(score < 0.75 for score in quality_scores)
        ):
            continue
        if any(
            context.get(field) is not True
            for field in (
                "directly_teaches_topic",
                "substantive",
                "factually_grounded",
                "self_contained",
                "is_standalone",
            )
        ):
            continue
        evidence_words = re.findall(
            r"[\w+#'-]+",
            str(context.get("topic_evidence_quote") or "").casefold(),
        )
        transcript_words = re.findall(
            r"[\w+#'-]+",
            str(row.get("transcript_snippet") or "").casefold(),
        )
        evidence_width = len(evidence_words)
        if (
            evidence_width < 5
            or evidence_width > 40
            or not any(
                transcript_words[index : index + evidence_width]
                == evidence_words
                for index in range(
                    max(0, len(transcript_words) - evidence_width + 1)
                )
            )
        ):
            continue
        if "surface_eligible" not in context:
            continue
        surface_eligible = context.get("surface_eligible")
        if isinstance(surface_eligible, str):
            surface_eligible = surface_eligible.strip().lower() in {
                "1", "true", "yes", "on",
            }
        surface_reason = str(context.get("surface_reason") or "").strip().lower()
        deferred_for_level = (
            not surface_eligible
            and surface_reason == "level_mismatch"
        )
        if not surface_eligible and not deferred_for_level:
            continue
        if not _search_context_has_usable_boundary(
            context, t_start=row.get("t_start"), t_end=row.get("t_end")
        ):
            continue
        verified_reusable_count += 1
    return verified_reusable_count > 0


def _generation_chain_analyzed_source_budget(
    conn,
    *,
    generation_id: str,
) -> int:
    """Return completed source analyses across a reusable generation chain."""
    generation_ids = _response_generation_ids(conn, generation_id)
    if not generation_ids:
        return 0

    placeholders = ", ".join("?" for _ in generation_ids)
    rows = fetch_all(
        conn,
        f"""
        SELECT jobs.id, jobs.usage_json, generations.generation_mode
        FROM reel_generations AS generations
        JOIN reel_generation_jobs AS jobs
          ON jobs.result_generation_id = generations.id
        WHERE generations.id IN ({placeholders})
          AND jobs.status IN ('completed', 'partial')
        """,
        tuple(generation_ids),
    )
    completed_sources = 0
    for row in rows:
        mode = str(row.get("generation_mode") or "").strip().lower()
        try:
            usage = json.loads(str(row.get("usage_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            usage = {}
        counters = usage.get("counters") if isinstance(usage, dict) else None
        if isinstance(counters, dict) and "analyzed_sources" in counters:
            try:
                completed_sources += max(
                    0, int(counters.get("analyzed_sources") or 0)
                )
            except (TypeError, ValueError, OverflowError):
                continue
        elif mode in GENERATION_SOURCE_BUDGETS:
            # Jobs written before actual-source accounting only recorded their
            # mode. Preserve their historical fixed-budget semantics.
            completed_sources += GENERATION_SOURCE_BUDGETS[mode]
    return completed_sources


def _generation_chain_consumed_video_ids(
    conn,
    *,
    generation_id: str,
) -> set[str]:
    """Return exact provider sources already selected across a generation chain."""
    generation_ids = _response_generation_ids(conn, generation_id)
    if not generation_ids:
        return set()

    placeholders = ", ".join("?" for _ in generation_ids)
    rows = fetch_all(
        conn,
        f"""
        SELECT jobs.usage_json
        FROM reel_generations AS generations
        JOIN reel_generation_jobs AS jobs
          ON jobs.result_generation_id = generations.id
        WHERE generations.id IN ({placeholders})
          AND jobs.status IN ('completed', 'partial', 'failed', 'exhausted')
        """,
        tuple(generation_ids),
    )
    consumed: set[str] = set()
    for row in rows:
        try:
            usage = json.loads(str(row.get("usage_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        raw_ids = usage.get("consumed_video_ids") if isinstance(usage, dict) else None
        if not isinstance(raw_ids, list):
            continue
        consumed.update(
            video_id
            for raw_id in raw_ids
            if (video_id := normalize_youtube_video_id(raw_id)) is not None
        )
    return consumed


def _generation_chain_failed_source_attempts(
    conn,
    *,
    generation_id: str,
) -> dict[str, int]:
    """Return durable failed analysis attempts across a generation chain."""
    generation_ids = _response_generation_ids(conn, generation_id)
    if not generation_ids:
        return {}

    placeholders = ", ".join("?" for _ in generation_ids)
    rows = fetch_all(
        conn,
        f"""
        SELECT jobs.usage_json
        FROM reel_generations AS generations
        JOIN reel_generation_jobs AS jobs
          ON jobs.result_generation_id = generations.id
        WHERE generations.id IN ({placeholders})
          AND jobs.status IN ('completed', 'partial', 'failed', 'exhausted')
        """,
        tuple(generation_ids),
    )
    attempts: dict[str, int] = {}
    for row in rows:
        try:
            usage = json.loads(str(row.get("usage_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        raw_attempts = (
            usage.get("failed_source_attempts")
            if isinstance(usage, dict)
            else None
        )
        if not isinstance(raw_attempts, dict):
            continue
        for raw_id, raw_count in raw_attempts.items():
            video_id = normalize_youtube_video_id(raw_id)
            if video_id is None:
                continue
            try:
                count = max(0, min(100, int(raw_count or 0)))
            except (TypeError, ValueError, OverflowError):
                continue
            attempts[video_id] = attempts.get(video_id, 0) + count
    return attempts


def _generation_chain_retryable_failed_source_ids(
    conn,
    *,
    generation_id: str,
) -> set[str]:
    """Return failed sources that still have one bounded analysis retry."""
    attempts = _generation_chain_failed_source_attempts(
        conn,
        generation_id=generation_id,
    )
    if not attempts:
        return set()
    consumed = _generation_chain_consumed_video_ids(
        conn,
        generation_id=generation_id,
    )
    return {
        video_id
        for video_id, count in attempts.items()
        if 0 < count < SOURCE_ANALYSIS_MAX_ATTEMPTS
        and video_id not in consumed
    }


def _generation_job_has_retryable_source_work(
    conn,
    job_row: dict[str, Any] | None,
) -> bool:
    """Return whether a terminal failure has bounded source work remaining."""
    if not job_row:
        return False
    if str(job_row.get("terminal_error_code") or "") in JOB_GLOBAL_PROVIDER_ERROR_CODES:
        return False
    generation_id = str(job_row.get("result_generation_id") or "").strip()
    if not generation_id:
        return False
    if _generation_chain_retryable_failed_source_ids(
        conn,
        generation_id=generation_id,
    ):
        return True
    if str(job_row.get("status") or "").strip() != "failed":
        return False
    try:
        usage = json.loads(str(job_row.get("usage_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        return False
    counters = usage.get("counters") if isinstance(usage, dict) else None
    try:
        return bool(
            isinstance(counters, dict)
            and int(counters.get("provider_cursor_open") or 0) > 0
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _generation_job_has_failed_source_attempts(
    conn,
    job_row: dict[str, Any] | None,
) -> bool:
    generation_id = str((job_row or {}).get("result_generation_id") or "").strip()
    return bool(
        generation_id
        and _generation_chain_failed_source_attempts(
            conn,
            generation_id=generation_id,
        )
    )


def _generation_job_allows_terminal_retry(
    conn,
    job_row: dict[str, Any] | None,
) -> bool:
    """Allow an explicit retry only for an unreleased terminal-empty batch."""
    if not job_row:
        return False
    status = str(job_row.get("status") or "").strip()
    error_code = str(job_row.get("terminal_error_code") or "").strip()
    if status != "exhausted" and not (
        status == "failed" and error_code in JOB_GLOBAL_PROVIDER_ERROR_CODES
    ):
        return False
    job_id = str(job_row.get("id") or "").strip()
    if job_id and _authoritative_job_release_reel_ids(conn, job_id):
        return False
    generation_id = str(job_row.get("result_generation_id") or "").strip()
    return not (
        generation_id
        and _authoritative_generation_release_reel_ids(conn, generation_id)
    )


def _failed_global_provider_terminal_retry_detail(
    conn,
    job_row: dict[str, Any] | None,
) -> dict[str, str] | None:
    if (
        str((job_row or {}).get("status") or "").strip() != "failed"
        or str((job_row or {}).get("terminal_error_code") or "").strip()
        not in JOB_GLOBAL_PROVIDER_ERROR_CODES
    ):
        return None
    job_id = str((job_row or {}).get("id") or "")
    if _generation_job_allows_terminal_retry(conn, job_row):
        return {
            "code": "terminal_retry_required",
            "message": (
                "Retry this failed reel batch explicitly before starting another "
                "generation."
            ),
            "terminal_job_id": job_id,
        }
    return {
        "code": "invalid_terminal_retry",
        "message": (
            "The failed reel batch has an authoritative release and cannot be restarted."
        ),
        "terminal_job_id": job_id,
    }


def _generation_chain_meets_source_budget(
    conn,
    *,
    generation_id: str,
    generation_mode: Literal["slow", "fast"],
) -> bool:
    return (
        _generation_chain_analyzed_source_budget(
            conn,
            generation_id=generation_id,
        )
        >= GENERATION_SOURCE_BUDGETS[generation_mode]
    )


def _canonical_excluded_video_id_set(values: object) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {
        video_id
        for value in values
        if (video_id := _bare_video_id(str(value or "")))
    }


def _generation_job_matches_request_params(
    job_row: dict[str, Any],
    request_params: dict[str, Any],
) -> bool:
    """Match the immutable dimensions of one learner generation stream."""
    prior_params = _job_request_params(job_row)
    expected_exclusions = _canonical_excluded_video_id_set(
        request_params.get("exclude_video_ids")
    )
    expected_relevance = _normalize_min_relevance(
        request_params.get("min_relevance")
    )
    expected_adaptation_fingerprint = str(
        request_params.get("adaptation_fingerprint")
        or GENERATION_EMPTY_ADAPTATION_FINGERPRINT
    )
    return bool(
        str(prior_params.get("request_schema_version") or "")
        == GENERATION_REQUEST_SCHEMA_VERSION
        and str(prior_params.get("knowledge_level") or "beginner")
        == str(request_params.get("knowledge_level") or "beginner")
        and str(prior_params.get("generation_mode") or "slow")
        == str(request_params.get("generation_mode") or "slow")
        and bool(prior_params.get("creative_commons_only"))
        == bool(request_params.get("creative_commons_only"))
        and _normalize_preferred_video_duration(
            str(prior_params.get("preferred_video_duration") or "any")
        )
        == _normalize_preferred_video_duration(
            str(request_params.get("preferred_video_duration") or "any")
        )
        and str(prior_params.get("language") or "en").strip().lower()
        == str(request_params.get("language") or "en").strip().lower()
        and str(prior_params.get("adaptation_fingerprint") or "")
        == expected_adaptation_fingerprint
        and _canonical_excluded_video_id_set(
            prior_params.get("exclude_video_ids")
        )
        == expected_exclusions
        and _normalize_min_relevance(prior_params.get("min_relevance"))
        == expected_relevance
    )


def _latest_compatible_generation_job(
    conn,
    *,
    material_id: str,
    learner_id: str,
    concept_id: str | None,
    content_fingerprint: str,
    request_params: dict[str, Any],
) -> dict[str, Any] | None:
    """Find the latest batch in the same learner/material/settings stream."""
    concept_clause = "AND concept_id IS NULL"
    query_params: tuple[Any, ...] = (
        material_id,
        learner_id,
        content_fingerprint,
    )
    if concept_id is not None:
        concept_clause = "AND concept_id = ?"
        query_params = (*query_params, concept_id)
    rows = fetch_all(
        conn,
        f"""
        SELECT *
        FROM reel_generation_jobs
        WHERE material_id = ?
          AND learner_id = ?
          AND content_fingerprint = ?
          {concept_clause}
          AND status IN ('queued', 'running', 'completed', 'partial', 'exhausted', 'failed')
        ORDER BY created_at DESC, id DESC
        LIMIT 50
        """,
        query_params,
    )
    for row in rows:
        if not _generation_job_matches_request_params(row, request_params):
            continue
        return row
    return None


def _latest_exact_generation_job(
    conn,
    *,
    material_id: str,
    learner_id: str,
    request_key: str,
) -> dict[str, Any] | None:
    """Return the latest non-cancelled job for one exact durable request."""
    return fetch_one(
        conn,
        """
        SELECT *
        FROM reel_generation_jobs
        WHERE material_id = ?
          AND learner_id = ?
          AND request_key = ?
          AND status IN ('queued', 'running', 'completed', 'partial', 'exhausted', 'failed')
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (material_id, learner_id, request_key),
    )


def _verified_cross_request_source_generation(
    conn,
    *,
    material_id: str,
    learner_id: str,
    request_key: str,
    concept_id: str | None,
    content_fingerprint: str,
    request_params: dict[str, Any],
    matched_request_params_out: dict[str, Any] | None = None,
) -> str | None:
    """Return compatible other-mode inventory only when its whole chain is verified."""
    if matched_request_params_out is not None:
        matched_request_params_out.clear()
    concept_clause = "AND concept_id IS NULL"
    candidate_params: tuple[Any, ...] = (
        material_id,
        learner_id,
        request_key,
        content_fingerprint,
    )
    if concept_id is not None:
        concept_clause = "AND concept_id = ?"
        candidate_params = (*candidate_params, concept_id)
    candidates = fetch_all(
        conn,
        f"""
        SELECT result_generation_id, request_params_json
        FROM reel_generation_jobs
        WHERE material_id = ?
          AND learner_id = ?
          AND request_key <> ?
          AND content_fingerprint = ?
          {concept_clause}
          AND status IN ('completed', 'partial')
          AND result_generation_id IS NOT NULL
          AND TRIM(result_generation_id) <> ''
        ORDER BY completed_at DESC, updated_at DESC, created_at DESC, id DESC
        LIMIT 20
        """,
        candidate_params,
    )
    expected_exclusions = _canonical_excluded_video_id_set(
        request_params.get("exclude_video_ids")
    )
    expected_relevance = _normalize_min_relevance(
        request_params.get("min_relevance")
    )
    cross_relevance_fallback: tuple[str, dict[str, Any]] | None = None
    for row in candidates:
        try:
            prior_params = json.loads(str(row.get("request_params_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(prior_params, dict):
            continue
        if (
            str(prior_params.get("request_schema_version") or "")
            != GENERATION_REQUEST_SCHEMA_VERSION
            or bool(prior_params.get("creative_commons_only"))
            != bool(request_params.get("creative_commons_only"))
            or _normalize_preferred_video_duration(
                str(prior_params.get("preferred_video_duration") or "any")
            )
            != _normalize_preferred_video_duration(
                str(request_params.get("preferred_video_duration") or "any")
            )
            or str(prior_params.get("language") or "en").strip().lower()
            != str(request_params.get("language") or "en").strip().lower()
            or not _canonical_excluded_video_id_set(
                prior_params.get("exclude_video_ids")
            ).issubset(expected_exclusions)
        ):
            continue
        generation_id = str(row.get("result_generation_id") or "").strip()
        if not generation_id:
            continue
        if _verified_reusable_generation_chain(
            conn,
            generation_id=generation_id,
            material_id=material_id,
        ):
            if (
                _normalize_min_relevance(prior_params.get("min_relevance"))
                == expected_relevance
            ):
                if matched_request_params_out is not None:
                    matched_request_params_out.update(prior_params)
                return generation_id
            if cross_relevance_fallback is None:
                cross_relevance_fallback = (generation_id, prior_params)
    if cross_relevance_fallback is None:
        return None
    generation_id, prior_params = cross_relevance_fallback
    if matched_request_params_out is not None:
        matched_request_params_out.update(prior_params)
    return generation_id


def _cancel_stale_active_adaptation_jobs(
    conn,
    *,
    material_id: str,
    learner_id: str,
    adaptation_fingerprint: str,
) -> None:
    current_knowledge_level = str(
        reel_service.learner_progress(conn, material_id, learner_id).get(
            "selected_level"
        )
        or "beginner"
    )
    rows = fetch_all(
        conn,
        "SELECT * FROM reel_generation_jobs "
        "WHERE material_id = ? AND learner_id = ? "
        "AND status IN ('queued', 'running')",
        (material_id, learner_id),
    )
    for row in rows:
        params = _job_request_params(row)
        if (
            str(params.get("request_schema_version") or "")
            == GENERATION_REQUEST_SCHEMA_VERSION
            and str(params.get("adaptation_fingerprint") or "")
            == adaptation_fingerprint
            and str(params.get("knowledge_level") or "beginner")
            == current_knowledge_level
        ):
            continue
        request_generation_cancellation(conn, job_id=str(row.get("id") or ""))


def _finalize_request_reel_order(
    conn,
    *,
    material_id: str,
    learner_id: str,
    rows: list[dict[str, Any]],
    previous_video_id: str,
    preserve_lesson_order_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Do not reapply legacy chronology to an already versioned feed order."""
    versioned_order = bool(rows) and all(
        bool(row.get("_selection_ordered")) for row in rows
    )
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for internal_key in [
            key
            for key in item
            if key.startswith("_selection_")
            and (
                not preserve_lesson_order_metadata
                or key not in _LESSON_ORDER_SELECTION_FIELDS
            )
        ]:
            item.pop(internal_key, None)
        cleaned.append(item)
    if versioned_order:
        return cleaned
    return reel_service.adaptive_curriculum_order(
        conn,
        material_id,
        learner_id,
        cleaned,
        previous_video_id=previous_video_id,
    )


def _learner_seen_reel_ids(conn, *, material_id: str, learner_id: str) -> set[str]:
    """Return durable clip-level history used by every feed surface."""
    try:
        rows = fetch_all(
            conn,
            """
            SELECT reel_id
            FROM learner_reel_progress
            WHERE learner_id = ?
              AND material_id = ?
              AND (scrolled_at IS NOT NULL OR completed_at IS NOT NULL)
            """,
            (str(learner_id or LEGACY_LEARNER_ID), material_id),
        )
    except Exception as exc:
        if "table" not in str(exc).lower() and "exist" not in str(exc).lower():
            raise
        return set()
    return {
        str(row.get("reel_id") or "").strip()
        for row in rows
        if str(row.get("reel_id") or "").strip()
    }


def _live_candidate_locked_prefix_ids(
    conn,
    *,
    material_id: str,
    learner_id: str,
    candidate_reel_ids: Iterable[str],
    job_started_at: object,
) -> list[str]:
    """Project current-job active/open membership onto durable emission order."""
    started_at = str(job_started_at or "").strip()
    emitted_ids = list(dict.fromkeys(
        reel_id
        for value in candidate_reel_ids
        if (reel_id := str(value or "").strip())
    ))[:LESSON_ORDER_MAX_CLIPS]
    if not started_at or not emitted_ids:
        return []
    placeholders = ", ".join("?" for _reel_id in emitted_ids)
    opened_ids = {
        str(row.get("reel_id") or "").strip()
        for row in fetch_all(
            conn,
            f"""
            SELECT reel_id
            FROM learner_reel_progress
            WHERE learner_id = ?
              AND material_id = ?
              AND updated_at >= ?
              AND max_fraction >= ?
              AND reel_id IN ({placeholders})
            """,
            (
                str(learner_id or LEGACY_LEARNER_ID),
                material_id,
                started_at,
                ACTIVE_REEL_OPEN_FRACTION,
                *emitted_ids,
            ),
        )
        if str(row.get("reel_id") or "").strip()
    }
    if not opened_ids:
        return []
    # Forward feed navigation is sequential. The furthest opened candidate proves
    # every earlier emitted candidate reached the immutable client prefix, even
    # when its fire-and-forget write committed later or was interrupted.
    last_opened_index = max(
        index
        for index, reel_id in enumerate(emitted_ids)
        if reel_id in opened_ids
    )
    return emitted_ids[: last_opened_index + 1]


def _generation_cursor_reel_count(
    conn,
    *,
    material_id: str,
    generation_id: str | None,
    reel_ids: set[str],
) -> int:
    """Count filtered rows from the active generation chain for page offsets."""
    clean_reel_ids = sorted(
        {
            str(reel_id or "").strip()
            for reel_id in reel_ids
            if str(reel_id or "").strip()
        }
    )
    generation_ids = _response_generation_ids(conn, generation_id)
    if not clean_reel_ids or not generation_ids:
        return 0
    reel_placeholders = ", ".join("?" for _reel_id in clean_reel_ids)
    generation_placeholders = ", ".join("?" for _generation_id in generation_ids)
    row = fetch_one(
        conn,
        f"""
        SELECT COUNT(DISTINCT id) AS reel_count
        FROM reels
        WHERE material_id = ?
          AND generation_id IN ({generation_placeholders})
          AND id IN ({reel_placeholders})
        """,
        (material_id, *generation_ids, *clean_reel_ids),
    )
    return max(0, int((row or {}).get("reel_count") or 0))


def _ranked_request_reels(
    conn,
    *,
    material_id: str,
    fast_mode: bool,
    generation_id: str | None,
    min_relevance: float | None,
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
    exclude_video_ids: list[str] | None = None,
    page: int = 1,
    limit: int = 5,
    learner_id: str = LEGACY_LEARNER_ID,
    exclude_reel_ids: list[str] | None = None,
    include_source_chain: bool = True,
    released_only: bool = False,
    preserve_lesson_order_metadata: bool = False,
    apply_generation_lesson_order: bool = True,
    seen_reel_ids_out: set[str] | None = None,
) -> list[dict[str, Any]]:
    try:
        material_row = fetch_one(conn, "SELECT subject_tag, source_type FROM materials WHERE id = ?", (material_id,))
    except (sqlite3.OperationalError, Exception) as _schema_exc:
        if "column" not in str(_schema_exc).lower() and "exist" not in str(_schema_exc).lower():
            raise
        material_row = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
    subject_tag = str((material_row or {}).get("subject_tag") or "").strip() or None
    strict_topic_only = str((material_row or {}).get("source_type") or "").strip().lower() == "topic"
    effective_min_relevance = _request_effective_min_relevance(
        min_relevance,
        page=page,
        subject_tag=subject_tag,
        strict_topic_only=strict_topic_only,
    )
    exclusions_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "video_ids": sorted({_bare_video_id(value) for value in (exclude_video_ids or []) if _bare_video_id(value)}),
                "reel_ids": sorted({str(value) for value in (exclude_reel_ids or []) if str(value)}),
                "usable_boundaries": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    content_fingerprint = _generation_content_fingerprint(
        conn,
        material_id=material_id,
        concept_id=None,
    )
    generation_ids = (
        _response_generation_ids(conn, generation_id)
        if include_source_chain
        else ([generation_id] if generation_id else [])
    )
    authoritative_release_ids = (
        _authoritative_release_reel_ids(conn, generation_id)
        if released_only and include_source_chain
        else None
    )
    authoritative_release_id_set = (
        set(authoritative_release_ids)
        if authoritative_release_ids is not None
        else None
    )
    difficulty_progress: dict[str, Any] | None = None
    if generation_ids:
        ranked_batches: list[list[dict[str, Any]]] = []
        for current_generation_id in generation_ids:
            if not current_generation_id:
                continue
            batch = reel_service.ranked_feed(
                conn,
                material_id,
                fast_mode=fast_mode,
                generation_id=current_generation_id,
                page_hint=page,
                learner_id=learner_id,
                exclusions_fingerprint=exclusions_fingerprint,
                content_fingerprint=content_fingerprint,
                require_verified_boundaries=True,
            )
            if authoritative_release_id_set is not None:
                batch = [
                    reel
                    for reel in batch
                    if str(reel.get("reel_id") or "").strip()
                    in authoritative_release_id_set
                ]
            ranked_batches.append(batch)
        if ranked_batches and all(
            str(reel.get("selection_contract_version") or "").strip()
            == SELECTION_CONTRACT_VERSION
            and bool(reel.get("_selection_ordered"))
            for batch in ranked_batches
            for reel in batch
        ):
            difficulty_progress = reel_service.learner_progress(
                conn, material_id, learner_id
            )
            target_stage = reel_service._selection_difficulty_stage(
                {
                    "difficulty": effective_level_target(
                        str(
                            difficulty_progress.get("selected_level")
                            or "beginner"
                        ),
                        float(
                            difficulty_progress.get("global_adjustment")
                            or 0.0
                        ),
                    )
                }
            )
            ranked = _merge_selection_ordered_reel_lists(
                *ranked_batches,
                target_stage=target_stage,
            )
        else:
            ranked = _merge_request_reel_lists(*ranked_batches)
    else:
        ranked = reel_service.ranked_feed(
            conn,
            material_id,
            fast_mode=fast_mode,
            generation_id=generation_id,
            page_hint=page,
            learner_id=learner_id,
            exclusions_fingerprint=exclusions_fingerprint,
            content_fingerprint=content_fingerprint,
            require_verified_boundaries=True,
        )
    selection_ordered = bool(ranked) and all(
        bool(reel.get("_selection_ordered")) for reel in ranked
    )
    if not selection_ordered and any(
        str(
            reel.get("_selection_contract_version")
            or reel.get("selection_contract_version")
            or ""
        ).strip()
        in reel_service.DIFFICULTY_FALLBACK_CONTRACTS
        for reel in ranked
    ):
        if difficulty_progress is None:
            difficulty_progress = reel_service.learner_progress(
                conn, material_id, learner_id
            )
        ranked = reel_service.select_difficulty_inventory(
            ranked,
            str(difficulty_progress.get("selected_level") or "beginner"),
            difficulty_target=effective_level_target(
                str(
                    difficulty_progress.get("selected_level") or "beginner"
                ),
                float(difficulty_progress.get("global_adjustment") or 0.0),
            ),
        )
    if authoritative_release_ids is not None:
        ranked = [
            reel
            for reel in ranked
            if str(reel.get("reel_id") or "").strip()
            in authoritative_release_id_set
        ]
    seen_reel_ids = _learner_seen_reel_ids(
        conn,
        material_id=material_id,
        learner_id=learner_id,
    )
    if seen_reel_ids_out is not None:
        seen_reel_ids_out.update(seen_reel_ids)
    if seen_reel_ids:
        ranked = [
            reel
            for reel in ranked
            if str(reel.get("reel_id") or "").strip() not in seen_reel_ids
        ]
    excluded_video_id_set = {
        _bare_video_id(video_id)
        for video_id in (exclude_video_ids or [])
        if _bare_video_id(video_id)
    }
    if excluded_video_id_set:
        ranked = [
            reel for reel in ranked
            if _bare_video_id(reel.get("video_id")) not in excluded_video_id_set
        ]
    previous_video_id = ""
    ordered_excluded_reel_ids = [
        str(reel_id) for reel_id in (exclude_reel_ids or []) if str(reel_id)
    ]
    if ordered_excluded_reel_ids:
        current_reel_id = ordered_excluded_reel_ids[-1]
        previous_video_id = next(
            (
                str(reel.get("video_id") or "")
                for reel in ranked
                if str(reel.get("reel_id") or "") == current_reel_id
            ),
            "",
        )
        if not previous_video_id:
            current_reel = fetch_one(
                conn,
                "SELECT video_id FROM reels WHERE id = ? AND material_id = ?",
                (current_reel_id, material_id),
            )
            previous_video_id = str((current_reel or {}).get("video_id") or "")
    excluded_reel_id_set = set(ordered_excluded_reel_ids)
    if excluded_reel_id_set:
        ranked = [
            reel for reel in ranked
            if str(reel.get("reel_id") or "") not in excluded_reel_id_set
        ]
    if authoritative_release_ids is None:
        if apply_generation_lesson_order:
            ranked = _apply_generation_lesson_order(
                conn,
                generation_id=generation_id,
                reels=ranked,
            )
    else:
        release_batches = [
            (current_generation_id, released_ids)
            for current_generation_id in generation_ids
            if (
                released_ids := _authoritative_generation_release_reel_ids(
                    conn,
                    current_generation_id,
                )
            )
        ]
        if release_batches and all(
            _stored_generation_lesson_order_ids(conn, current_generation_id)
            == released_ids
            for current_generation_id, released_ids in release_batches
        ):
            by_id = {
                str(reel.get("reel_id") or "").strip(): reel
                for reel in ranked
                if str(reel.get("reel_id") or "").strip()
            }
            ranked = [
                by_id[reel_id]
                for reel_id in authoritative_release_ids
                if reel_id in by_id
            ]

    if page <= 1:
        shaped = _shape_request_page_reels(
            ranked,
            page=page,
            limit=limit,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
            min_relevance=effective_min_relevance,
            preferred_video_duration=preferred_video_duration,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        return _finalize_request_reel_order(
            conn,
            material_id=material_id,
            learner_id=learner_id,
            rows=shaped,
            previous_video_id=previous_video_id,
            preserve_lesson_order_metadata=preserve_lesson_order_metadata,
        )

    cumulative_batches: list[list[dict[str, Any]]] = []
    for current_page in range(1, max(1, int(page)) + 1):
        current_min_relevance = _request_effective_min_relevance(
            min_relevance,
            page=current_page,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
        )
        cumulative_batches.append(
            _shape_request_page_reels(
                ranked,
                page=current_page,
                limit=limit,
                subject_tag=subject_tag,
                strict_topic_only=strict_topic_only,
                min_relevance=current_min_relevance,
                preferred_video_duration=preferred_video_duration,
                target_clip_duration_sec=target_clip_duration_sec,
                target_clip_duration_min_sec=target_clip_duration_min_sec,
                target_clip_duration_max_sec=target_clip_duration_max_sec,
            )
        )
    return _finalize_request_reel_order(
        conn,
        material_id=material_id,
        learner_id=learner_id,
        rows=_merge_request_reel_lists(*cumulative_batches),
        previous_video_id=previous_video_id,
        preserve_lesson_order_metadata=preserve_lesson_order_metadata,
    )


def _bare_video_id(value: str | None) -> str:
    """Strip a platform prefix (`yt:`/`ig:`/`tt:`) → the bare source id.

    Clip-engine reels persist with a `yt:<id>`-prefixed DB video_id while legacy
    rows and every client use the BARE 11-char id (derived from video_url).
    Normalizing both sides at comparison time keeps client pagination exclusion
    working across legacy + clip-engine rows.
    """
    return str(value or "").strip().split(":", 1)[-1]


def _normalize_excluded_video_ids(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values or []:
        clean = str(raw_value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def _parse_excluded_video_ids_param(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    # Cap raw input length and the resulting list to keep a malicious client
    # from sending a megabyte of exclusion IDs.
    if len(raw_value) > 16_000:
        raw_value = raw_value[:16_000]
    normalized = _normalize_excluded_video_ids(raw_value.split(","))
    if len(normalized) > 500:
        normalized = normalized[-500:]
    return normalized


def _parse_excluded_reel_ids_param(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    if len(raw_value) > 32_000:
        raw_value = raw_value[:32_000]
    normalized = _normalize_excluded_video_ids(raw_value.split(","))
    return normalized[:200]


def _generation_db_transaction_definitely_aborted(exc: BaseException) -> bool:
    """Return whether PostgreSQL guarantees that a failed transaction rolled back."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        sqlstate = str(
            getattr(current, "sqlstate", "")
            or getattr(current, "pgcode", "")
            or ""
        ).strip().upper()
        if sqlstate in {"40001", "40P01"}:
            return True
        for chained in (current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False


def _run_generation_db_transaction(
    step: str,
    work: Callable[[Any], Any],
    *,
    retry_should_stop: Callable[[], bool] | None = None,
    replay_after_unknown_commit: bool = False,
) -> Any:
    """Retry one failed PostgreSQL transaction on a fresh connection.

    The conservative default avoids replaying an unknown commit. Callers may opt
    in only when every write has stable identities that converge on replay.
    """
    for attempt in range(1, GENERATION_DB_TRANSACTION_MAX_ATTEMPTS + 1):
        body_completed = False
        try:
            with get_conn(transactional=True) as conn:
                result = work(conn)
                body_completed = True
            return result
        except Exception as exc:
            if (
                attempt >= GENERATION_DB_TRANSACTION_MAX_ATTEMPTS
                or isinstance(
                    exc,
                    (DatabaseIntegrityError, HTTPException, ValueError),
                )
                or not is_transient_postgres_transaction_error(exc)
                or (
                    body_completed
                    and not _generation_db_transaction_definitely_aborted(exc)
                    and not replay_after_unknown_commit
                )
            ):
                raise
            if retry_should_stop is not None:
                try:
                    retry_stopped = retry_should_stop()
                except Exception as stop_exc:
                    # Python implicitly links an exception raised by the guard
                    # to the transaction exception currently being handled.
                    # Do not let that outer transient context make a permanent
                    # guard failure look retryable.
                    stop_context = stop_exc.__context__
                    if stop_context is exc:
                        stop_exc.__context__ = None
                    try:
                        stop_check_is_transient = (
                            is_transient_postgres_transaction_error(stop_exc)
                        )
                    finally:
                        stop_exc.__context__ = stop_context
                    if not stop_check_is_transient:
                        raise
                    logger.warning(
                        "generation retry lease check failed transiently; "
                        "continuing bounded retry step=%s error=%s",
                        step,
                        (str(stop_exc).splitlines() or [stop_exc.__class__.__name__])[0][
                            :180
                        ],
                    )
                else:
                    if retry_stopped:
                        raise
            logger.warning(
                "retrying generation database step after transient PostgreSQL "
                "failure step=%s attempt=%d/%d error=%s",
                step,
                attempt,
                GENERATION_DB_TRANSACTION_MAX_ATTEMPTS,
                (str(exc).splitlines() or [exc.__class__.__name__])[0][:180],
            )
    raise AssertionError("unreachable")


def _persist_generation_provider_usage(
    job_id: str,
    record: ProviderUsageRecord,
    *,
    retry_should_stop: Callable[[], bool] | None = None,
) -> None:
    """Store every provider response instead of treating one rate-limit hit as a generation."""
    usage_id = str(uuid.uuid4())

    def persist(conn: Any) -> None:
        record_provider_usage(
            conn,
            job_id=job_id,
            provider=record.provider,
            operation=record.operation,
            model=record.model_used or None,
            billable_requests=record.billable_requests,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            metadata={
                **record.metadata,
                "attempt": record.attempt,
                "status_code": record.status_code,
                "quality_degraded": record.quality_degraded,
                "error_code": record.error_code,
                "timestamp": record.timestamp,
            },
            usage_id=usage_id,
        )

    _run_generation_db_transaction(
        "provider_usage",
        persist,
        retry_should_stop=retry_should_stop,
    )


def _generation_gemini_ledger_exposure(
    conn: Any,
    job_id: str,
    *,
    prior_records: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Reconcile Gemini exposure and Groq usage from one durable ledger scan."""
    database_records: list[dict[str, Any]] = []
    durable_ticket_ids: set[str] = set()
    for row in fetch_all(
        conn,
        """
        SELECT id, provider, operation, model, billable_requests, input_tokens,
               output_tokens, total_tokens, metadata_json
        FROM generation_provider_usage
        WHERE job_id = ? AND provider IN ('gemini', 'groq')
        ORDER BY created_at, id
        """,
        (job_id,),
    ):
        try:
            metadata = json.loads(str(row.get("metadata_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        safe_metadata = metadata if isinstance(metadata, dict) else {}
        usage_id = str(row.get("id") or "")
        provider = str(row.get("provider") or "").strip().casefold()
        if (
            provider == "gemini"
            and usage_id
            and safe_metadata.get("ticket_schema_version")
            == "gemini_dispatch_ticket_v1"
        ):
            durable_ticket_ids.add(usage_id)
        database_records.append({
            "id": usage_id,
            "provider": provider,
            "operation": str(row.get("operation") or ""),
            "model_used": str(row.get("model") or ""),
            "attempt": safe_metadata.get("attempt"),
            "timestamp": safe_metadata.get("timestamp"),
            "status_code": safe_metadata.get("status_code"),
            "billable_requests": int(row.get("billable_requests") or 0),
            "input_tokens": int(row.get("input_tokens") or 0),
            "output_tokens": int(row.get("output_tokens") or 0),
            "total_tokens": int(row.get("total_tokens") or 0),
            "quality_degraded": bool(safe_metadata.get("quality_degraded")),
            "error_code": str(safe_metadata.get("error_code") or ""),
            "metadata": safe_metadata,
        })

    records: list[dict[str, Any]] = []
    for prior_record in prior_records:
        if not isinstance(prior_record, dict):
            continue
        prior = dict(prior_record)
        raw_metadata = prior.get("metadata")
        prior_metadata = (
            raw_metadata if isinstance(raw_metadata, Mapping) else {}
        )
        prior_ticket_id = str(
            prior.get("id")
            or prior_metadata.get("gemini_ticket_id")
            or ""
        )
        if (
            str(prior.get("provider") or "").casefold() == "gemini"
            and prior_ticket_id
            and prior_ticket_id in durable_ticket_ids
        ):
            continue
        records.append(prior)

    durable_records: list[dict[str, Any]] = []
    for record in database_records:
        metadata = record.get("metadata")
        if (
            str(record.get("provider") or "").casefold() == "gemini"
            and isinstance(metadata, Mapping)
            and metadata.get("ticket_schema_version")
            == "gemini_dispatch_ticket_v1"
        ):
            durable_records.append(record)
        else:
            records.append(record)

    def provider_record_identity(
        record: Mapping[str, Any],
        provider: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        raw_metadata = record.get("metadata")
        metadata = (
            dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        envelope: dict[str, Any] = {}
        for field_name in (
            "attempt",
            "timestamp",
            "status_code",
            "quality_degraded",
            "error_code",
        ):
            envelope[field_name] = (
                record.get(field_name)
                if record.get(field_name) is not None
                else metadata.get(field_name)
            )
            metadata.pop(field_name, None)
        identity = json.dumps(
            {
                "provider": provider,
                "operation": str(record.get("operation") or ""),
                "model_used": str(record.get("model_used") or ""),
                "billable_requests": int(
                    record.get("billable_requests") or 0
                ),
                "input_tokens": int(record.get("input_tokens") or 0),
                "output_tokens": int(record.get("output_tokens") or 0),
                "total_tokens": int(record.get("total_tokens") or 0),
                "envelope": envelope,
                "metadata": metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return identity, metadata, envelope

    exposure = gemini_usage_records_exposure(records)
    committed_cost = max(
        0.0,
        float(exposure.get("committed_cost_usd") or 0.0),
    )
    unknown_cost = max(
        0.0,
        float(
            exposure.get("billing_unknown_cost_exposure_usd")
            or 0.0
        ),
    )
    lifetime_reserved_cost = max(
        0.0,
        float(
            exposure.get("lifetime_reserved_worst_case_cost_usd")
            or 0.0
        ),
    )
    unknown_calls = 0
    unknown_attempts = 0
    seen_non_ticket_records: set[str] = set()
    for record in records:
        if str(record.get("provider") or "").casefold() != "gemini":
            continue
        record_key, safe_metadata, _envelope = (
            provider_record_identity(record, "gemini")
        )
        if record_key in seen_non_ticket_records:
            continue
        seen_non_ticket_records.add(record_key)
        raw_unknown_attempts = safe_metadata.get("billing_unknown_attempts")
        try:
            record_unknown_attempts = max(
                0,
                int(raw_unknown_attempts or 0),
            )
        except (TypeError, ValueError, OverflowError):
            record_unknown_attempts = 0
        if record_unknown_attempts:
            unknown_calls += 1
            unknown_attempts += record_unknown_attempts

    def nonnegative_cost(value: object, *, fallback: float = 0.0) -> float:
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError):
            return fallback
        return (
            normalized
            if math.isfinite(normalized) and normalized >= 0.0
            else fallback
        )

    for record in durable_records:
        metadata = dict(record.get("metadata") or {})
        admitted_cost = nonnegative_cost(
            metadata.get("admitted_cost_usd"),
        )
        lifetime_reserved_cost += admitted_cost
        state = str(metadata.get("ticket_state") or "admitted")
        if state == "released":
            continue
        if state == "settled_known":
            committed_cost += nonnegative_cost(
                metadata.get("actual_cost_usd"),
            )
            continue
        ticket_unknown_cost = (
            nonnegative_cost(
                metadata.get("billing_unknown_reserved_cost_usd"),
                fallback=admitted_cost,
            )
            if state == "settled_unknown"
            else admitted_cost
        )
        committed_cost += ticket_unknown_cost
        unknown_cost += ticket_unknown_cost
        if ticket_unknown_cost > 1e-9:
            unknown_calls += 1
            unknown_attempts += 1

    groq_records: list[dict[str, Any]] = []
    seen_groq_records: set[str] = set()
    for record in sorted(
        records,
        key=lambda candidate: 0 if candidate.get("id") else 1,
    ):
        if str(record.get("provider") or "").casefold() != "groq":
            continue
        structural_key, metadata, envelope = provider_record_identity(
            record,
            "groq",
        )
        dispatch_id = str(
            metadata.get("groq_dispatch_id") or ""
        ).strip()
        record_key = (
            f"groq_dispatch:{dispatch_id}"
            if dispatch_id
            else structural_key
        )
        if record_key in seen_groq_records:
            continue
        seen_groq_records.add(record_key)
        normalized_record = {
            key: value
            for key, value in record.items()
            if key != "id"
        }
        normalized_record["metadata"] = {
            **metadata,
            **{
                field_name: field_value
                for field_name, field_value in envelope.items()
                if (
                    field_value is not None
                    and normalized_record.get(field_name) is None
                )
            },
        }
        groq_records.append(normalized_record)

    if unknown_cost <= 1e-9:
        unknown_calls = 0
        unknown_attempts = 0

    groq_context = GenerationContext("fast")
    for record in groq_records:
        groq_context.record(
            ProviderUsageRecord(
                provider="groq",
                operation=str(
                    record.get("operation") or "transcript"
                ),
                attempt=max(1, int(record.get("attempt") or 1)),
                timestamp=str(record.get("timestamp") or ""),
                status_code=record.get("status_code"),
                billable_requests=max(
                    0,
                    int(record.get("billable_requests") or 0),
                ),
                input_tokens=max(0, int(record.get("input_tokens") or 0)),
                output_tokens=max(
                    0,
                    int(record.get("output_tokens") or 0),
                ),
                total_tokens=max(0, int(record.get("total_tokens") or 0)),
                model_used=str(record.get("model_used") or ""),
                quality_degraded=bool(record.get("quality_degraded")),
                error_code=str(record.get("error_code") or ""),
                metadata=dict(record.get("metadata") or {}),
            ),
            persist=False,
        )
    groq_payload = groq_context.usage_payload()
    groq_summary = dict(groq_payload.get("summary") or {})
    groq_by_stage = dict(groq_payload.get("by_stage") or {})

    return {
        "committed_cost_usd": committed_cost,
        "cost_exposure_usd": committed_cost,
        "billing_unknown_cost_exposure_usd": unknown_cost,
        "lifetime_reserved_worst_case_cost_usd": lifetime_reserved_cost,
        "durable_ticket_count": float(len(durable_records)),
        "billing_unknown_calls": float(unknown_calls),
        "billing_unknown_attempts": float(unknown_attempts),
        "groq_calls": float(groq_summary.get("groq_calls") or 0),
        "groq_attempts": float(groq_summary.get("groq_attempts") or 0),
        "groq_audio_seconds": float(
            groq_summary.get("groq_audio_seconds") or 0.0
        ),
        "groq_billed_audio_seconds": float(
            groq_summary.get("groq_billed_audio_seconds") or 0.0
        ),
        "groq_known_billed_cost_usd": float(
            groq_summary.get("groq_known_billed_cost_usd") or 0.0
        ),
        "groq_billing_unknown_calls": float(
            groq_summary.get("groq_billing_unknown_calls") or 0
        ),
        "groq_billing_unknown_attempts": float(
            groq_summary.get("groq_billing_unknown_attempts") or 0
        ),
        "groq_billing_unknown_reserved_cost_usd": float(
            groq_summary.get(
                "groq_billing_unknown_reserved_cost_usd"
            )
            or 0.0
        ),
        "groq_provider_records": groq_records,
        "groq_by_stage": groq_by_stage,
    }


def _job_request_params(job_row: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(str(job_row.get("request_params_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _continuation_delivered_reel_ids(
    conn,
    continuation_token: str | None,
) -> list[str]:
    """Return authoritative reel ids already delivered by prior batches."""
    job_id = str(continuation_token or "").strip()
    if not job_id:
        return []
    job_row = get_generation_job(conn, job_id)
    generation_id = str((job_row or {}).get("result_generation_id") or "").strip()
    if not generation_id:
        return []
    generation_rows = _generation_chain_rows_snapshot(conn, (generation_id,))
    generation_ids = _snapshot_generation_ids(generation_rows, generation_id)
    releases = _authoritative_generation_releases_snapshot(
        conn,
        generation_ids,
    )
    delivered = _authoritative_release_ids_from_snapshot(
        generation_ids=generation_ids,
        generation_rows=generation_rows,
        releases_by_generation=releases,
    )
    return list(dict.fromkeys(delivered or ()))


def _currently_surfaceable_generation_reel_ids(
    conn,
    job_row: dict[str, Any],
) -> set[str]:
    """Revalidate persisted inventory with the current ranked-feed guards."""
    generation_id = str(job_row.get("result_generation_id") or "").strip()
    material_id = str(job_row.get("material_id") or "").strip()
    if not generation_id or not material_id:
        return set()
    params = _job_request_params(job_row)
    fast_mode = str(params.get("generation_mode") or "slow") == "fast"
    learner_id = str(job_row.get("learner_id") or LEGACY_LEARNER_ID)
    content_fingerprint = _generation_content_fingerprint(
        conn,
        material_id=material_id,
        concept_id=(str(job_row.get("concept_id") or "").strip() or None),
    )
    valid_ids: set[str] = set()
    for current_generation_id in _response_generation_ids(conn, generation_id):
        current = reel_service.ranked_feed(
            conn,
            material_id,
            fast_mode=fast_mode,
            generation_id=current_generation_id,
            learner_id=learner_id,
            content_fingerprint=content_fingerprint,
            require_verified_boundaries=True,
        )
        valid_ids.update(
            str(reel.get("reel_id") or "").strip()
            for reel in current
            if str(reel.get("reel_id") or "").strip()
        )
    return valid_ids


def _generation_job_reels(
    conn,
    job_row: dict[str, Any],
    *,
    requested_override: int | None = None,
    organizer_candidate_limit: int | None = None,
    apply_release_order: bool = True,
    preserve_lesson_order_metadata: bool = False,
    prior_unseen_reels_out: list[dict[str, Any]] | None = None,
    editorial_excluded_reel_ids: Iterable[str] = (),
    editorial_guard_reel_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    generation_id = str(job_row.get("result_generation_id") or "").strip()
    if not generation_id:
        return []
    params = _job_request_params(job_row)
    continuation_token = str(params.get("continuation_token") or "").strip()
    delivered_reel_ids = _continuation_delivered_reel_ids(
        conn,
        continuation_token,
    )
    editorial_excluded_ids = list(dict.fromkeys(
        reel_id
        for value in editorial_excluded_reel_ids
        if (reel_id := str(value or "").strip())
    ))
    editorial_guard_ids = {
        reel_id
        for value in editorial_guard_reel_ids
        if (reel_id := str(value or "").strip())
    }
    seen_reel_ids: set[str] = set()
    mode = "fast" if str(params.get("generation_mode") or "slow") == "fast" else "slow"
    requested = max(
        1,
        min(
            GENERATION_OUTPUT_CEILINGS[mode],
            int(
                requested_override
                if requested_override is not None
                else params.get("num_reels") or GENERATION_OUTPUT_CEILINGS[mode]
            ),
        ),
    )
    stored_order_ids = _stored_generation_lesson_order_ids(conn, generation_id)
    internal_candidate_limit = (
        requested
        if organizer_candidate_limit is None
        else max(
            1,
            min(LESSON_ORDER_MAX_CLIPS, int(organizer_candidate_limit)),
        )
    )
    ranking_limit = (
        max(internal_candidate_limit, len(stored_order_ids or []))
        if apply_release_order
        else internal_candidate_limit
    )
    released_only = str(job_row.get("status") or "").strip() in {
        "completed",
        "partial",
        "exhausted",
    }
    ranked = _ranked_request_reels(
        conn,
        material_id=str(job_row.get("material_id") or ""),
        fast_mode=mode == "fast",
        generation_id=generation_id,
        min_relevance=_normalize_min_relevance(params.get("min_relevance")),
        preferred_video_duration=_normalize_preferred_video_duration(
            str(params.get("preferred_video_duration") or "any")
        ),
        target_clip_duration_sec=0,
        target_clip_duration_min_sec=None,
        target_clip_duration_max_sec=None,
        exclude_video_ids=list(params.get("exclude_video_ids") or []),
        page=1,
        limit=ranking_limit,
        learner_id=str(job_row.get("learner_id") or LEGACY_LEARNER_ID),
        exclude_reel_ids=(
            [] if prior_unseen_reels_out is not None else delivered_reel_ids
        ),
        include_source_chain=True,
        released_only=released_only,
        preserve_lesson_order_metadata=preserve_lesson_order_metadata,
        apply_generation_lesson_order=apply_release_order,
        **(
            {"seen_reel_ids_out": seen_reel_ids}
            if editorial_excluded_ids and editorial_guard_ids
            else {}
        ),
    )
    internal_reels = [
        _public_generation_reel(
            reel,
            preserve_lesson_order_metadata=True,
        )
        for reel in ranked
    ]
    valid_reels = _current_selection_contract_reels(internal_reels)
    boundary_statuses = {
        str(reel.get("_selection_boundary_status") or "")
        .strip()
        .casefold()
        for reel in valid_reels
    }
    valid_reel_ids = {
        str(reel.get("reel_id") or "").strip()
        for reel in valid_reels
        if str(reel.get("reel_id") or "").strip()
    }
    needs_source_release_protection = bool(
        prior_unseen_reels_out is not None
        or set(editorial_excluded_ids).intersection(valid_reel_ids)
        or (
            "best_effort" in boundary_statuses
            and bool(
                boundary_statuses.intersection({"verified", "context_aligned"})
            )
        )
    )
    source_release_reel_ids: list[str] = []
    if needs_source_release_protection:
        source_generation_id = str(
            job_row.get("source_generation_id") or ""
        ).strip()
        if not source_generation_id:
            generation_row = _fetch_generation_row(conn, generation_id) or {}
            source_generation_id = str(
                generation_row.get("source_generation_id") or ""
            ).strip()
        if source_generation_id:
            # A release is not a view. Preserve its recursive editorial plan;
            # healthy single-lane batches avoid this parent-chain lookup.
            source_release_reel_ids = list(
                _authoritative_release_reel_ids(conn, source_generation_id) or ()
            )
    prior_release_reel_ids = list(dict.fromkeys([
        *delivered_reel_ids,
        *source_release_reel_ids,
    ]))
    available_guard_ids = {
        str(reel.get("reel_id") or "").strip()
        for reel in valid_reels
        if str(reel.get("reel_id") or "").strip()
    } | set(delivered_reel_ids) | seen_reel_ids
    if (
        editorial_excluded_ids
        and editorial_guard_ids
        and editorial_guard_ids.issubset(available_guard_ids)
    ):
        editorial_excluded_id_set = set(editorial_excluded_ids)
        valid_reels = [
            reel
            for reel in valid_reels
            if (
                (reel_id := str(reel.get("reel_id") or "").strip())
                not in editorial_excluded_id_set
                or reel_id in prior_release_reel_ids
            )
        ]
    valid_reels = _prefer_proven_boundary_inventory(
        valid_reels,
        protected_reel_ids=prior_release_reel_ids,
    )
    if prior_unseen_reels_out is not None and prior_release_reel_ids:
        prior_release_reel_id_set = set(prior_release_reel_ids)
        valid_reels_by_id = {
            reel_id: reel
            for reel in valid_reels
            if (reel_id := str(reel.get("reel_id") or "").strip())
        }
        prior_unseen_reels_out.extend(
            valid_reels_by_id[reel_id]
            for reel_id in prior_release_reel_ids
            if reel_id in valid_reels_by_id
        )
        valid_reels = [
            reel
            for reel in valid_reels
            if str(reel.get("reel_id") or "").strip()
            not in prior_release_reel_id_set
        ]
    ordered_reels = (
        _apply_generation_lesson_order(
            conn,
            generation_id=generation_id,
            reels=valid_reels,
        )
        if apply_release_order
        else valid_reels
    )
    # A persisted organizer decision is the authoritative editorial subset.
    # ``num_reels`` controls acquisition cadence, but must not truncate a
    # larger already-verified subset selected from the bounded candidate pool.
    selected = ordered_reels[:ranking_limit]
    if preserve_lesson_order_metadata:
        return selected
    return [_public_generation_reel(reel) for reel in selected]


def _reused_generation_reels(
    conn,
    *,
    generation_id: str,
    material_id: str,
    concept_id: str | None,
    learner_id: str,
    request_params: dict[str, Any],
    requested: int,
) -> list[dict[str, Any]]:
    """Read a compatible reservoir with difficulty retained as a soft signal."""
    return _generation_job_reels(
        conn,
        {
            "result_generation_id": generation_id,
            "material_id": material_id,
            "concept_id": concept_id,
            "learner_id": learner_id,
            "request_params_json": json.dumps(request_params),
        },
        requested_override=requested,
    )


def _current_level_reusable_generation_reel_count(
    conn,
    *,
    generation_id: str | None,
    material_id: str,
    concept_id: str | None,
    learner_id: str,
    request_params: dict[str, Any],
    requested: int,
    editorial_excluded_reel_ids: Iterable[str] = (),
    editorial_guard_reel_ids: Iterable[str] = (),
) -> int:
    """Count reusable source-chain inventory up to the startup target."""
    if not generation_id:
        return 0
    return min(
        INITIAL_READY_REEL_TARGET,
        len(
            _generation_job_reels(
                conn,
                {
                    "result_generation_id": generation_id,
                    "material_id": material_id,
                    "concept_id": concept_id,
                    "learner_id": learner_id,
                    "request_params_json": json.dumps(request_params),
                },
                requested_override=INITIAL_READY_REEL_TARGET,
                organizer_candidate_limit=INITIAL_READY_REEL_TARGET,
                apply_release_order=False,
                editorial_excluded_reel_ids=editorial_excluded_reel_ids,
                editorial_guard_reel_ids=editorial_guard_reel_ids,
            )
        ),
    )


def _generation_usage_with_authoritative_release_count(
    usage: object,
    *,
    released_reels: int,
) -> dict[str, Any]:
    """Attach final release accounting without changing acquisition counters."""
    normalized_usage = dict(usage) if isinstance(usage, Mapping) else {}
    raw_summary = normalized_usage.get("summary")
    summary = dict(raw_summary) if isinstance(raw_summary, Mapping) else {}
    try:
        released_count = max(0, int(released_reels))
    except (TypeError, ValueError, OverflowError):
        released_count = 0
    known_cost = max(
        0.0,
        float(summary.get("known_billed_cost_usd") or 0.0),
        float(summary.get("estimated_cost_usd") or 0.0),
        float(summary.get("telemetry_priced_cost_usd") or 0.0),
    )
    billing_unknown = bool(
        int(summary.get("billing_unknown_calls") or 0)
        or float(summary.get("billing_unknown_reserved_cost_usd") or 0.0)
        > 1e-9
    )
    summary["released_reels"] = released_count
    summary["cost_per_released_reel_usd"] = (
        round(known_cost / released_count, 8)
        if released_count and not billing_unknown
        else None
    )
    normalized_usage["summary"] = summary
    return normalized_usage


def _generation_usage_with_authoritative_gemini_exposure(
    conn: Any,
    job_row: Mapping[str, Any],
    usage: object,
) -> dict[str, Any]:
    """Overlay late Gemini settlements and crash-surviving Groq usage."""
    normalized_usage = dict(usage) if isinstance(usage, Mapping) else {}
    job_id = str(job_row.get("id") or "")
    if (
        conn is None
        or not job_id
        or str(job_row.get("status") or "")
        not in GENERATION_TERMINAL_STATUSES
    ):
        return normalized_usage

    raw_provider_calls = normalized_usage.get("provider_calls")
    prior_records = (
        [
            dict(record)
            for record in raw_provider_calls
            if isinstance(record, dict)
        ]
        if isinstance(raw_provider_calls, list)
        else []
    )
    exposure = _generation_gemini_ledger_exposure(
        conn,
        job_id,
        prior_records=prior_records,
    )
    ledger_groq_calls = max(
        0,
        int(exposure.get("groq_calls") or 0),
    )
    raw_summary = normalized_usage.get("summary")
    summary = dict(raw_summary) if isinstance(raw_summary, Mapping) else {}
    stored_groq_claim = any(
        float(summary.get(field_name) or 0.0) > 0.0
        for field_name in (
            "groq_calls",
            "groq_attempts",
            "groq_audio_seconds",
            "groq_billed_audio_seconds",
            "groq_known_billed_cost_usd",
            "groq_billing_unknown_calls",
            "groq_billing_unknown_attempts",
            "groq_billing_unknown_reserved_cost_usd",
        )
    )
    if (
        int(exposure.get("durable_ticket_count") or 0) <= 0
        and ledger_groq_calls <= 0
        and not stored_groq_claim
    ):
        return normalized_usage
    committed_cost = max(
        0.0,
        float(exposure.get("committed_cost_usd") or 0.0),
    )
    unknown_cost = min(
        committed_cost,
        max(
            0.0,
            float(
                exposure.get("billing_unknown_cost_exposure_usd")
                or 0.0
            ),
        ),
    )
    known_cost = max(0.0, committed_cost - unknown_cost)
    unknown_calls = max(
        0,
        int(exposure.get("billing_unknown_calls") or 0),
    )
    unknown_attempts = max(
        0,
        int(exposure.get("billing_unknown_attempts") or 0),
    )
    lifetime_reserved_cost = max(
        0.0,
        float(
            exposure.get("lifetime_reserved_worst_case_cost_usd")
            or 0.0
        ),
    )

    raw_budget = normalized_usage.get("budget")
    budget = dict(raw_budget) if isinstance(raw_budget, Mapping) else {}
    raw_gemini_budget = budget.get("gemini")
    gemini_budget = (
        dict(raw_gemini_budget)
        if isinstance(raw_gemini_budget, Mapping)
        else {}
    )
    gemini_budget.update(
        {
            "reserved_cost_usd": lifetime_reserved_cost,
            "lifetime_reserved_worst_case_cost_usd": lifetime_reserved_cost,
            "settled_cost_exposure_usd": committed_cost,
            "committed_cost_usd": committed_cost,
            "billing_unknown_cost_exposure_usd": unknown_cost,
            "cost_exposure_usd": committed_cost,
        }
    )
    budget["gemini"] = gemini_budget
    normalized_usage["budget"] = budget

    groq_known_cost = max(
        0.0,
        float(exposure.get("groq_known_billed_cost_usd") or 0.0),
    )
    groq_unknown_cost = max(
        0.0,
        float(
            exposure.get("groq_billing_unknown_reserved_cost_usd")
            or 0.0
        ),
    )
    groq_unknown_calls = max(
        0,
        int(exposure.get("groq_billing_unknown_calls") or 0),
    )
    groq_unknown_attempts = max(
        0,
        int(exposure.get("groq_billing_unknown_attempts") or 0),
    )
    total_known_cost = known_cost + groq_known_cost
    total_unknown_cost = unknown_cost + groq_unknown_cost
    summary.update(
        {
            "gemini_known_billed_cost_usd": known_cost,
            "gemini_billing_unknown_calls": unknown_calls,
            "gemini_billing_unknown_attempts": unknown_attempts,
            "gemini_billing_unknown_reserved_cost_usd": unknown_cost,
            "groq_calls": ledger_groq_calls,
            "groq_attempts": max(
                0,
                int(exposure.get("groq_attempts") or 0),
            ),
            "groq_audio_seconds": max(
                0.0,
                float(exposure.get("groq_audio_seconds") or 0.0),
            ),
            "groq_billed_audio_seconds": max(
                0.0,
                float(
                    exposure.get("groq_billed_audio_seconds")
                    or 0.0
                ),
            ),
            "groq_known_billed_cost_usd": groq_known_cost,
            "groq_billing_unknown_calls": groq_unknown_calls,
            "groq_billing_unknown_attempts": groq_unknown_attempts,
            "groq_billing_unknown_reserved_cost_usd": (
                groq_unknown_cost
            ),
            "estimated_cost_usd": total_known_cost,
            "known_billed_cost_usd": total_known_cost,
            "telemetry_priced_cost_usd": total_known_cost,
            "current_cost_exposure_usd": (
                committed_cost + groq_known_cost + groq_unknown_cost
            ),
            "billing_unknown_reserved_cost_usd": total_unknown_cost,
            "billing_unknown_calls": unknown_calls + groq_unknown_calls,
            "billing_unknown_attempts": (
                unknown_attempts + groq_unknown_attempts
            ),
            "reserved_worst_case_cost_usd": lifetime_reserved_cost,
            "lifetime_reserved_worst_case_cost_usd": lifetime_reserved_cost,
        }
    )
    accepted_clips = max(0, int(summary.get("accepted_clips") or 0))
    summary["cost_per_accepted_clip_usd"] = (
        round(total_known_cost / accepted_clips, 8)
        if accepted_clips and total_unknown_cost <= 1e-9
        else None
    )
    normalized_usage["summary"] = summary

    ledger_groq_records = exposure.get("groq_provider_records")
    if isinstance(ledger_groq_records, list):
        normalized_usage["provider_calls"] = [
            record
            for record in prior_records
            if str(record.get("provider") or "").casefold() != "groq"
        ] + [
            dict(record)
            for record in ledger_groq_records
            if isinstance(record, dict)
        ]

    ledger_groq_by_stage = exposure.get("groq_by_stage")
    if isinstance(ledger_groq_by_stage, Mapping):
        raw_by_stage = normalized_usage.get("by_stage")
        by_stage = (
            {
                str(stage): dict(bucket)
                for stage, bucket in raw_by_stage.items()
                if isinstance(bucket, Mapping)
            }
            if isinstance(raw_by_stage, Mapping)
            else {}
        )
        groq_stage_names = {"groq_boundary_asr"}
        for record in prior_records:
            if str(record.get("provider") or "").casefold() != "groq":
                continue
            metadata = record.get("metadata")
            safe_metadata = (
                metadata if isinstance(metadata, Mapping) else {}
            )
            groq_stage_names.add(str(
                safe_metadata.get("stage")
                or record.get("operation")
                or "unknown"
            ))
        groq_stage_names.update(
            str(stage) for stage in ledger_groq_by_stage
        )
        for stage in groq_stage_names:
            by_stage.pop(stage, None)
        for stage, ledger_bucket in ledger_groq_by_stage.items():
            if not isinstance(ledger_bucket, Mapping):
                continue
            by_stage[str(stage)] = dict(ledger_bucket)
        normalized_usage["by_stage"] = by_stage
    if "released_reels" in summary:
        normalized_usage = _generation_usage_with_authoritative_release_count(
            normalized_usage,
            released_reels=summary["released_reels"],
        )
    return normalized_usage


def _generation_job_status_payload(conn, job_row: dict[str, Any]) -> dict[str, Any]:
    try:
        usage = json.loads(str(job_row.get("usage_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        usage = {}
    usage = _generation_usage_with_authoritative_gemini_exposure(
        conn,
        job_row,
        usage,
    )
    error_code = str(job_row.get("terminal_error_code") or "").strip()
    error_message = str(job_row.get("terminal_error_message") or "").strip()
    error: dict[str, Any] | None = None
    if error_code or error_message:
        error = {"code": error_code or "generation_failed", "message": error_message}
        try:
            detail = json.loads(str(job_row.get("terminal_error_json") or "null"))
        except (TypeError, json.JSONDecodeError):
            detail = None
        if detail is not None:
            error["detail"] = detail
    status = str(job_row.get("status") or "")
    surfaceable_terminal = status in {"completed", "partial", "exhausted"}
    result_generation_id = str(job_row.get("result_generation_id") or "").strip()
    return {
        "job_id": str(job_row.get("id") or ""),
        "status": status or "queued",
        "phase": str(job_row.get("phase") or ""),
        "progress": float(job_row.get("progress") or 0.0),
        "attempt_count": int(job_row.get("attempt_count") or 0),
        "max_attempts": int(job_row.get("max_attempts") or 3),
        "lease_expires_at": _normalize_datetime_for_api(job_row.get("lease_expires_at")),
        "heartbeat_at": _normalize_datetime_for_api(job_row.get("heartbeat_at")),
        "deadline_at": _normalize_datetime_for_api(job_row.get("deadline_at")),
        "material_id": str(job_row.get("material_id") or ""),
        "request_key": str(job_row.get("request_key") or ""),
        "result_generation_id": result_generation_id or None,
        "model_used": str(job_row.get("model_used") or "") or None,
        "quality_degraded": bool(job_row.get("quality_degraded")),
        "usage": usage if isinstance(usage, dict) else {},
        "error": error,
        "reels": _generation_job_reels(conn, job_row) if surfaceable_terminal else [],
        "reconciliation_tail_reel_ids": (
            _stored_generation_reconciliation_tail_ids(
                conn,
                result_generation_id,
            )
            if surfaceable_terminal and result_generation_id
            else None
        ),
        "created_at": _normalize_datetime_for_api(job_row.get("created_at")),
        "started_at": _normalize_datetime_for_api(job_row.get("started_at")),
        "completed_at": _normalize_datetime_for_api(job_row.get("completed_at")),
    }


def _sanitize_generation_replay_events(
    conn,
    job_row: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    authoritative_reels: list[dict[str, Any]] | None = None
    job_status = str((job_row or {}).get("status") or "")
    surfaceable_terminal = bool(
        job_row
        and job_status in {"completed", "partial", "exhausted"}
    )
    suppress_inventory = job_status in {"failed", "cancelled"}
    if job_row and job_status in GENERATION_TERMINAL_STATUSES:
        authoritative_reels = (
            _generation_job_reels(conn, job_row)
            if (
                conn is not None
                and surfaceable_terminal
                and not suppress_inventory
            )
            else []
        )
    params = _job_request_params(job_row or {})
    mode = (
        "fast"
        if str(params.get("generation_mode") or "slow") == "fast"
        else "slow"
    )
    try:
        candidate_limit = max(
            1,
            min(
                GENERATION_OUTPUT_CEILINGS[mode],
                int(
                    params.get("num_reels")
                    or GENERATION_OUTPUT_CEILINGS[mode]
                ),
            ),
        )
    except (TypeError, ValueError, OverflowError):
        candidate_limit = GENERATION_OUTPUT_CEILINGS[mode]
    candidate_reels = [
        payload.get("reel")
        for event in events
        if str(event.get("type") or "") == "candidate"
        and isinstance((payload := event.get("payload")), dict)
        and payload.get("provisional") is True
        and isinstance(payload.get("reel"), dict)
    ]
    usable_candidate_ids = (
        _usable_boundary_reel_ids(
            conn,
            [
                str(reel.get("reel_id") or "")
                for reel in candidate_reels
            ],
        )
        if conn is not None and candidate_reels
        else set()
    )
    authoritative_reel_ids = {
        str(reel.get("reel_id") or "").strip()
        for reel in authoritative_reels or []
        if str(reel.get("reel_id") or "").strip()
    }
    authoritative_clip_keys = {
        clip_key
        for reel in authoritative_reels or []
        if _reel_source_video_id(reel)
        for _reel_id, clip_key in [_reel_identity_key(reel)]
    }
    emitted_candidate_ids: set[str] = set()
    emitted_candidate_clip_keys: set[str] = set()
    for raw_event in events:
        event = dict(raw_event)
        event_type = str(event.get("type") or "")
        payload = dict(event.get("payload") or {})
        if event_type == "candidate":
            reel = payload.get("reel")
            if (
                payload.get("provisional") is not True
                or not isinstance(reel, dict)
            ):
                continue
            reel_id, clip_key = _reel_identity_key(reel)
            meaningful_clip_key = (
                clip_key if _reel_source_video_id(reel) else ""
            )
            if (
                not reel_id
                or reel_id not in usable_candidate_ids
                or reel_id in emitted_candidate_ids
                or (
                    meaningful_clip_key
                    and meaningful_clip_key in emitted_candidate_clip_keys
                )
                or len(emitted_candidate_ids) >= candidate_limit
            ):
                continue
            if authoritative_reels is not None and (
                reel_id not in authoritative_reel_ids
                and (
                    not meaningful_clip_key
                    or meaningful_clip_key not in authoritative_clip_keys
                )
            ):
                continue
            emitted_candidate_ids.add(reel_id)
            if meaningful_clip_key:
                emitted_candidate_clip_keys.add(meaningful_clip_key)
            payload["reel"] = _public_generation_reel(reel)
            payload["provisional"] = True
            event["payload"] = payload
        if event_type == "final" and job_row is not None:
            if suppress_inventory:
                authoritative_reels = []
            elif authoritative_reels is None:
                authoritative_reels = _generation_job_reels(
                    conn,
                    job_row if surfaceable_terminal else {**job_row, "status": "completed"},
                )
            payload["reels"] = authoritative_reels
            if suppress_inventory:
                payload["generation_id"] = None
                payload["reconciliation_tail_reel_ids"] = None
            else:
                payload["reconciliation_tail_reel_ids"] = (
                    _stored_generation_reconciliation_tail_ids(
                        conn,
                        str(job_row.get("result_generation_id") or "").strip(),
                    )
                )
            payload["authoritative"] = True
            event["payload"] = payload
        elif event_type == "terminal" and job_row is not None:
            payload["usage"] = _generation_usage_with_authoritative_gemini_exposure(
                conn,
                job_row,
                payload.get("usage"),
            )
            event["payload"] = payload
        sanitized.append(event)
    return sanitized


def _generation_job_db_should_stop(
    job_id: str,
    lease_owner: str,
    *,
    now: datetime | None = None,
) -> bool:
    checked_at = now or datetime.now(timezone.utc)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    checked_at = checked_at.astimezone(timezone.utc)
    with get_conn() as conn:
        current = get_generation_job(conn, job_id)
    if not current:
        return True
    if str(current.get("status") or "") != "running":
        return True
    if str(current.get("lease_owner") or "") != lease_owner:
        return True
    if int(current.get("cancel_requested") or 0):
        return True

    def reached(value: object, *, missing_is_expired: bool) -> bool:
        normalized = _normalize_datetime_for_api(value)
        if not normalized:
            return missing_is_expired
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return checked_at >= parsed.astimezone(timezone.utc)

    return reached(current.get("lease_expires_at"), missing_is_expired=True) or reached(
        current.get("deadline_at"), missing_is_expired=False
    )


def _generation_exhaustion_message(counters: dict[str, int]) -> str:
    """Describe the furthest successful pipeline stage without blaming captions."""
    discovered = int(counters.get("discovered_videos") or 0)
    transcripts = int(counters.get("usable_transcripts") or 0)
    transcript_failures = int(counters.get("transcript_failures") or 0)
    transcript_timeouts = int(counters.get("transcript_timeouts") or 0)
    clip_fetch_timeouts = int(counters.get("clip_fetch_timeouts") or 0)
    timeouts = transcript_timeouts + clip_fetch_timeouts
    gemini_empty_results = int(counters.get("gemini_empty_results") or 0)
    topic_rejections = int(counters.get("topic_rejections") or 0)

    if discovered <= 0:
        return "No matching YouTube videos were discovered for this topic."
    if transcripts <= 0:
        if clip_fetch_timeouts > 0:
            return (
                "Matching YouTube videos were discovered, but transcript and clip "
                "analysis did not complete before the generation deadline."
            )
        if transcript_timeouts > 0:
            return (
                "Matching YouTube videos were discovered, but no usable timestamped "
                "transcripts were available before the generation deadline."
            )
        if transcript_failures > 0:
            return (
                "Matching YouTube videos were discovered, but none provided a usable "
                "timestamped transcript."
            )
        return (
            "Matching YouTube videos were discovered, but transcript retrieval did not "
            "complete."
        )
    if timeouts > 0 and topic_rejections > 0:
        return (
            "Some video analyses did not finish before the generation deadline; clips "
            "from the completed timestamped transcripts did not pass the topic and "
            "quality checks."
        )
    if timeouts > 0 and gemini_empty_results > 0:
        return (
            "Some video analyses did not finish before the generation deadline; the "
            "completed videos produced no clips that passed the content quality checks."
        )
    if timeouts > 0:
        return (
            "Some video analyses did not finish before the generation deadline, and "
            "the completed videos produced no valid clips."
        )
    if topic_rejections > 0:
        return (
            "Videos with usable timestamped transcripts were found, but no clips passed "
            "the topic and quality checks."
        )
    if gemini_empty_results > 0:
        return (
            "Videos with usable timestamped transcripts were found, but no clips passed "
            "the content quality checks."
        )
    return "No discovered YouTube videos produced valid clips."


def _learner_concept_signals(
    conn,
    *,
    material_id: str,
    learner_id: str,
    propagate_concept_families: bool = False,
    candidate_concept_ids: set[str] | None = None,
    remediation_concept_ids_out: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    exact_remediation_ids = (
        [] if remediation_concept_ids_out is not None else None
    )
    coverage, adjustments, _, _ = reel_service._learner_adaptation_context(
        conn,
        material_id,
        learner_id,
        propagate_concept_families=propagate_concept_families,
        candidate_concept_ids=candidate_concept_ids,
        remediation_concept_ids_out=exact_remediation_ids,
    )
    if remediation_concept_ids_out is not None:
        remediation_concept_ids_out.extend(exact_remediation_ids or [])
    return {
        concept_key: {
            "helpful": float(
                coverage.get(concept_key, {}).get("helpful", 0.0)
            ),
            "confusing": float(
                coverage.get(concept_key, {}).get("confusing", 0.0)
            ),
            "adjustment": float(adjustments.get(concept_key, 0.0)),
        }
        for concept_key in (set(coverage) | set(adjustments))
    }


def _learner_signal_reel_ids(
    conn,
    *,
    material_id: str,
    learner_id: str,
) -> list[str]:
    """Return recent reel evidence behind nonzero feedback and quiz signals."""
    clean_learner_id = str(learner_id or LEGACY_LEARNER_ID)
    signal_rows: list[tuple[str, int, str]] = []
    for row in fetch_all(
        conn,
        """
        SELECT f.reel_id,
               COALESCE(f.mastery_updated_at, f.updated_at, f.created_at, '') AS signal_at
        FROM reel_feedback f
        JOIN reels r ON r.id = f.reel_id
        WHERE f.learner_id = ?
          AND r.material_id = ?
          AND (f.helpful <> 0 OR f.confusing <> 0)
        ORDER BY signal_at DESC
        LIMIT ?
        """,
        (clean_learner_id, material_id, LESSON_SIGNAL_HISTORY_LIMIT),
    ):
        reel_id = str(row.get("reel_id") or "").strip()
        if reel_id:
            signal_rows.append((str(row.get("signal_at") or ""), 0, reel_id))
    try:
        assessment_rows = fetch_all(
            conn,
            """
            SELECT source_reel_id AS reel_id,
                   COALESCE(created_at, '') AS signal_at
            FROM assessment_concept_outcomes
            WHERE learner_id = ?
              AND material_id = ?
              AND adjustment <> 0
              AND source_reel_id IS NOT NULL
            ORDER BY signal_at DESC
            LIMIT ?
            """,
            (clean_learner_id, material_id, LESSON_SIGNAL_HISTORY_LIMIT),
        )
    except sqlite3.OperationalError as exc:
        if "no such table: assessment_concept_outcomes" not in str(exc):
            raise
        assessment_rows = []
    for row in assessment_rows:
        reel_id = str(row.get("reel_id") or "").strip()
        if reel_id:
            signal_rows.append((str(row.get("signal_at") or ""), 1, reel_id))
    latest_signal_by_reel: dict[str, tuple[str, int]] = {}
    for signal_at, source_rank, reel_id in signal_rows:
        signal_key = (signal_at, source_rank)
        if signal_key > latest_signal_by_reel.get(reel_id, ("", -1)):
            latest_signal_by_reel[reel_id] = signal_key
    ordered = sorted(
        latest_signal_by_reel,
        key=lambda reel_id: (*latest_signal_by_reel[reel_id], reel_id),
    )
    return ordered[-LESSON_SIGNAL_HISTORY_LIMIT:]


def _lesson_prior_coverage(
    conn,
    *,
    material_id: str,
    reel_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical and recent clip-level history in release order."""
    clean_ids = list(dict.fromkeys(
        clean_id
        for reel_id in reel_ids
        if (clean_id := str(reel_id or "").strip())
    ))
    if not clean_ids:
        return [], []
    release_rank_by_id = {
        reel_id: release_rank
        for release_rank, reel_id in enumerate(clean_ids)
    }
    rows: list[dict[str, Any]] = []
    # Chunk only for database parameter limits; never truncate semantic history.
    for offset in range(0, len(clean_ids), 400):
        reel_id_chunk = clean_ids[offset:offset + 400]
        placeholders = ", ".join("?" for _reel_id in reel_id_chunk)
        rows.extend(fetch_all(
            conn,
            f"""
            SELECT reels.id AS reel_id, reels.concept_id,
                   reels.search_context_json, reels.t_start,
                   reels.t_end, reels.ai_summary, reels.transcript_snippet,
                   concepts.title AS concept_title
            FROM reels
            LEFT JOIN concepts ON concepts.id = reels.concept_id
            WHERE reels.material_id = ?
              AND reels.id IN ({placeholders})
            """,
            (material_id, *reel_id_chunk),
        ))
    rows.sort(key=lambda row: release_rank_by_id.get(
        str(row.get("reel_id") or ""),
        len(release_rank_by_id),
    ))
    coverage: dict[tuple[str, str], dict[str, Any]] = {}
    recent_objectives: list[dict[str, Any]] = []
    for row in rows:
        if has_incompatible_gemini_concept_family_contract(
            row.get("search_context_json")
        ):
            continue
        metadata = reel_service._selection_metadata(
            row.get("search_context_json"),
            t_start=row.get("t_start"),
            t_end=row.get("t_end"),
        )
        concept_id = str(row.get("concept_id") or "").strip()
        concept_family = " ".join(
            str(metadata.get("_selection_concept_family") or "").split()
        )[:96]
        concept_title = " ".join(
            str(row.get("concept_title") or "").split()
        )[:240]
        objective_parts = (
            " ".join(str(row.get("ai_summary") or "").split()),
            " ".join(str(row.get("transcript_snippet") or "").split()),
        )
        learning_objective_excerpt = next(
            (part[:500] for part in objective_parts if part),
            "",
        )
        identity = (
            ("concept", concept_id)
            if concept_id
            else ("family", concept_family.casefold())
        )
        if not identity[1]:
            continue
        item = coverage.setdefault(
            identity,
            {
                "concept_id": concept_id,
                "concept_family": concept_family,
                "concept_title": concept_title,
                "learning_objective_excerpts": [],
                "delivered_count": 0,
            },
        )
        item["delivered_count"] = int(item["delivered_count"]) + 1
        objective_excerpts = item["learning_objective_excerpts"]
        if (
            learning_objective_excerpt
            and learning_objective_excerpt not in objective_excerpts
        ):
            objective_excerpts.append(learning_objective_excerpt)
            del objective_excerpts[:-3]
        if learning_objective_excerpt:
            recent_objectives.append({
                "concept_id": concept_id,
                "concept_family": concept_family,
                "concept_title": concept_title,
                "learning_objective_excerpt": learning_objective_excerpt,
                "release_rank": release_rank_by_id.get(
                    str(row.get("reel_id") or ""),
                    0,
                ),
            })
        obligation_keys = intent_obligation_keys(
            metadata.get("_selection_intent_obligations") or ()
        )
        if obligation_keys:
            item["intent_obligation_keys"] = sorted({
                *item.get("intent_obligation_keys", ()),
                *obligation_keys,
            })
    return [coverage[key] for key in sorted(coverage)], recent_objectives


def _lesson_prior_concept_coverage(
    conn,
    *,
    material_id: str,
    reel_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Compatibility view for canonical prior-concept coverage callers."""
    concepts, _recent_objectives = _lesson_prior_coverage(
        conn,
        material_id=material_id,
        reel_ids=reel_ids,
    )
    return concepts


def _learner_adaptation_fingerprint(
    conn,
    *,
    material_id: str,
    learner_id: str,
) -> str:
    concept_signals = _learner_concept_signals(
        conn,
        material_id=material_id,
        learner_id=learner_id,
        # Gemini-created target facets can appear while a job is running. Its
        # stale-input fingerprint therefore stays keyed to signal sources only.
        propagate_concept_families=False,
    )
    return hashlib.sha256(
        json.dumps(
            concept_signals,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _lock_learner_adaptation(
    conn,
    *,
    material_id: str,
    learner_id: str,
) -> None:
    """Serialize final release with feedback and completed assessments."""
    material = fetch_one(
        conn,
        "SELECT knowledge_level FROM materials WHERE id = ?",
        (material_id,),
    )
    if not material:
        raise ValueError(f"unknown material_id: {material_id}")
    timestamp = now_iso()
    execute_modify(
        conn,
        "INSERT INTO learner_material_progress "
        "(learner_id, material_id, selected_level, global_adjustment, "
        "difficulty_reset_at, feedback_revision, updated_at) "
        "VALUES (?, ?, ?, 0.0, ?, 0, ?) "
        "ON CONFLICT(learner_id, material_id) DO NOTHING",
        (
            learner_id,
            material_id,
            str(material.get("knowledge_level") or "beginner"),
            "" if learner_id == LEGACY_LEARNER_ID else timestamp,
            timestamp,
        ),
    )
    execute_modify(
        conn,
        "UPDATE learner_material_progress SET updated_at = updated_at "
        "WHERE learner_id = ? AND material_id = ?",
        (learner_id, material_id),
    )


def _run_leased_generation_job(
    job_row: dict[str, Any],
    worker_stop: threading.Event | None = None,
) -> None:
    job_id = str(job_row.get("id") or "")
    lease_owner = str(job_row.get("lease_owner") or "")
    durable_attempt_count = max(1, int(job_row.get("attempt_count") or 1))
    params = _job_request_params(job_row)
    is_continuation = bool(str(params.get("continuation_token") or "").strip())
    has_fresh_source_budget = bool(params.get("fresh_source_budget"))
    mode: Literal["fast", "slow"] = "fast" if params.get("generation_mode") == "fast" else "slow"
    requested_count = max(
        1,
        min(
            GENERATION_OUTPUT_CEILINGS[mode],
            int(params.get("num_reels") or GENERATION_OUTPUT_CEILINGS[mode]),
        ),
    )
    lesson_candidate_limit = LESSON_ORDER_CANDIDATE_LIMITS[mode]
    material_id = str(job_row.get("material_id") or "")
    concept_id = str(job_row.get("concept_id") or "") or None
    learner_id = str(job_row.get("learner_id") or LEGACY_LEARNER_ID)
    local_stop = threading.Event()
    active_worker_stop = worker_stop or _generation_worker_stop

    def heartbeat_loop() -> None:
        while not local_stop.wait(GENERATION_HEARTBEAT_SEC):
            try:
                with get_conn(transactional=True) as heartbeat_conn:
                    alive = heartbeat_generation_job(
                        heartbeat_conn,
                        job_id=job_id,
                        lease_owner=lease_owner,
                        lease_seconds=GENERATION_LEASE_SEC,
                    )
                if not alive:
                    local_stop.set()
                    return
            except Exception:
                logger.exception("generation heartbeat failed job_id=%s", job_id)

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        name=f"generation-heartbeat-{job_id[:8]}",
        daemon=True,
    )

    last_cancel_check = 0.0
    cached_cancelled = False

    def should_cancel() -> bool:
        nonlocal last_cancel_check, cached_cancelled
        if local_stop.is_set() or active_worker_stop.is_set():
            return True
        now_mono = time.monotonic()
        if now_mono - last_cancel_check >= 0.5:
            last_cancel_check = now_mono
            cached_cancelled = _generation_job_db_should_stop(job_id, lease_owner)
        return cached_cancelled

    def db_retry_should_stop() -> bool:
        return bool(
            local_stop.is_set()
            or active_worker_stop.is_set()
            or _generation_job_db_should_stop(job_id, lease_owner)
        )

    def reserve_gemini_ticket(
        *,
        ticket_id: str,
        operation: str,
        model: str,
        reservation: Mapping[str, Any],
    ) -> object:
        reservation_metadata = dict(reservation)
        reservation_id = max(
            1,
            int(
                reservation_metadata.get("gemini_reservation_id")
                or 1
            ),
        )
        return _run_generation_db_transaction(
            "gemini_ticket_admission",
            lambda ticket_conn: admit_gemini_dispatch_ticket(
                ticket_conn,
                ticket_id=ticket_id,
                job_id=job_id,
                lease_owner=lease_owner,
                expected_attempt_count=durable_attempt_count,
                operation=operation,
                model=model,
                admitted_cost_usd=float(
                    reservation_metadata.get("admitted_cost_usd")
                    or 0.0
                ),
                provider_attempt=reservation_id,
                reservation_id=reservation_id,
                reserved_input_tokens=max(
                    0,
                    int(
                        reservation_metadata.get("reserved_input_tokens")
                        or 0
                    ),
                ),
                reserved_output_tokens=max(
                    1,
                    int(
                        reservation_metadata.get("reserved_output_tokens")
                        or 1
                    ),
                ),
                reserved_cost_usd=float(
                    reservation_metadata.get("reserved_cost_usd")
                    or 0.0
                ),
                billing_cost_multiplier=float(
                    reservation_metadata.get("billing_cost_multiplier")
                    or 1.0
                ),
                admitted_physical_attempts=max(
                    1,
                    int(
                        reservation_metadata.get(
                            "admitted_physical_attempts"
                        )
                        or 1
                    ),
                ),
                metadata={
                    **reservation_metadata,
                    "gemini_ticket_id": ticket_id,
                    "gemini_durable_ticket": True,
                },
            ),
            retry_should_stop=db_retry_should_stop,
            replay_after_unknown_commit=True,
        )

    def settle_gemini_ticket(
        *,
        ticket_id: str,
        state: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        actual_cost_usd: float | None = None,
        unknown_cost_usd: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> object:
        settlement_metadata = dict(metadata or {})
        if model:
            settlement_metadata["model_used"] = model
        return _run_generation_db_transaction(
            "gemini_ticket_settlement",
            lambda ticket_conn: settle_gemini_dispatch_ticket(
                ticket_conn,
                ticket_id=ticket_id,
                job_id=job_id,
                state=state,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                actual_cost_usd=(
                    actual_cost_usd if state == "settled_known" else None
                ),
                unknown_cost_usd=(
                    unknown_cost_usd if state == "settled_unknown" else None
                ),
                metadata=settlement_metadata,
            ),
            # Settlement is keyed only by ticket and job. It must converge
            # after the admitting worker loses its lease to a successor.
            replay_after_unknown_commit=True,
        )

    context = GenerationContext(
        mode,
        generation_id=job_id,
        usage_sink=lambda record: _persist_generation_provider_usage(
            job_id,
            record,
            retry_should_stop=db_retry_should_stop,
        ),
        gemini_ticket_reserve_sink=reserve_gemini_ticket,
        gemini_ticket_settle_sink=settle_gemini_ticket,
        cache_store=DatabaseProviderCache(),
        require_acoustic_boundaries=True,
    )
    generation_id = ""
    setup_generation_candidate_id = (
        str(job_row.get("result_generation_id") or "").strip() or str(uuid.uuid4())
    )
    try:
        previous_usage = json.loads(str(job_row.get("usage_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        previous_usage = {}
    if not isinstance(previous_usage, dict):
        previous_usage = {}
    raw_previous_provider_calls = previous_usage.get("provider_calls")
    previous_provider_calls = (
        [dict(record) for record in raw_previous_provider_calls if isinstance(record, dict)]
        if isinstance(raw_previous_provider_calls, list)
        else []
    )
    context.budget.restore_gemini_retry_exposure(previous_usage.get("budget"))
    if durable_attempt_count > 1:
        ledger_exposure = _run_generation_db_transaction(
            "provider_usage_recovery",
            lambda usage_conn: _generation_gemini_ledger_exposure(
                usage_conn,
                job_id,
                prior_records=previous_provider_calls,
            ),
            retry_should_stop=db_retry_should_stop,
        )
        context.budget.restore_gemini_retry_exposure({
            "mode": mode,
            "gemini": ledger_exposure,
        })
        recovered_groq_records = ledger_exposure.get(
            "groq_provider_records"
        )
        if isinstance(recovered_groq_records, list):
            previous_provider_calls = [
                record
                for record in previous_provider_calls
                if str(record.get("provider") or "").casefold() != "groq"
            ] + [
                dict(record)
                for record in recovered_groq_records
                if isinstance(record, dict)
            ]
    previous_consumed_source_ids = {
        video_id
        for raw_id in previous_usage.get("consumed_video_ids", [])
        if (video_id := normalize_youtube_video_id(raw_id)) is not None
    } if isinstance(previous_usage.get("consumed_video_ids"), list) else set()
    previous_capacity_deferred_source_ids = {
        video_id
        for raw_id in previous_usage.get("capacity_deferred_video_ids", [])
        if (video_id := normalize_youtube_video_id(raw_id)) is not None
    } if isinstance(previous_usage.get("capacity_deferred_video_ids"), list) else set()
    previous_failed_source_attempts: dict[str, int] = {}
    raw_previous_failures = previous_usage.get("failed_source_attempts")
    if isinstance(raw_previous_failures, dict):
        for raw_id, raw_count in raw_previous_failures.items():
            video_id = normalize_youtube_video_id(raw_id)
            if video_id is None:
                continue
            try:
                count = max(0, min(100, int(raw_count or 0)))
            except (TypeError, ValueError, OverflowError):
                continue
            if count:
                previous_failed_source_attempts[video_id] = count
    raw_previous_counters = previous_usage.get("counters")
    previous_counters = (
        raw_previous_counters if isinstance(raw_previous_counters, dict) else {}
    )
    raw_retry_errors = previous_usage.get("retry_errors")
    previous_retry_errors = (
        [dict(error) for error in raw_retry_errors if isinstance(error, dict)]
        if isinstance(raw_retry_errors, list)
        else []
    )
    completed_source_ids: set[str] = set(previous_consumed_source_ids)
    attempted_source_ids: set[str] = set()
    capacity_deferred_source_ids: set[str] = set(
        previous_capacity_deferred_source_ids
    )
    retrieved_video_ids: set[str] = set()

    def generation_usage_payload(
        *,
        retry_error: dict[str, Any] | None = None,
        terminal: bool = False,
    ) -> dict[str, Any]:
        if terminal:
            context.budget.finalize_gemini_exposure()
        payload = context.usage_payload(
            prior_records=previous_provider_calls,
        )

        previous_summary = previous_usage.get("summary")
        if isinstance(previous_summary, dict):
            summary = dict(payload.get("summary") or {})
            provider_summary_fields = set(summary) - {
                "accepted_clips",
                "fallback_reasons",
                "rejection_reason_counts",
                "rejected_boundaries",
            }
            non_additive_summary_fields = {
                "completion_cost_limit_usd",
                "cost_limit_usd",
                "cost_per_accepted_clip_usd",
                "current_cost_exposure_usd",
                "lifetime_reserved_worst_case_cost_usd",
                "reserved_worst_case_cost_usd",
            } | provider_summary_fields
            for field, previous_value in previous_summary.items():
                current_value = summary.get(field)
                if field == "fallback_reasons":
                    previous_reasons = (
                        [
                            str(value)
                            for value in previous_value
                            if str(value).strip()
                        ]
                        if isinstance(previous_value, list)
                        else []
                    )
                    current_reasons = (
                        [
                            str(value)
                            for value in current_value
                            if str(value).strip()
                        ]
                        if isinstance(current_value, list)
                        else []
                    )
                    summary[field] = list(dict.fromkeys([
                        *previous_reasons,
                        *current_reasons,
                    ]))
                elif field == "rejection_reason_counts":
                    counts = (
                        dict(current_value)
                        if isinstance(current_value, dict)
                        else {}
                    )
                    if isinstance(previous_value, dict):
                        for reason, raw_count in previous_value.items():
                            try:
                                counts[str(reason)] = int(counts.get(str(reason)) or 0) + int(
                                    raw_count or 0
                                )
                            except (TypeError, ValueError, OverflowError):
                                continue
                    summary[field] = dict(sorted(counts.items()))
                elif (
                    field not in non_additive_summary_fields
                    and isinstance(previous_value, (int, float))
                    and not isinstance(previous_value, bool)
                    and isinstance(current_value, (int, float))
                    and not isinstance(current_value, bool)
                ):
                    summary[field] = previous_value + current_value
                elif field not in summary:
                    summary[field] = previous_value
            accepted_clips = int(summary.get("accepted_clips") or 0)
            billing_unknown_calls = int(summary.get("billing_unknown_calls") or 0)
            summary["cost_per_accepted_clip_usd"] = (
                round(float(summary.get("estimated_cost_usd") or 0.0) / accepted_clips, 8)
                if accepted_clips and not billing_unknown_calls
                else None
            )
            payload["summary"] = summary

        previous_attempt_budgets = previous_usage.get("attempt_budgets")
        attempt_budgets = (
            [
                dict(budget)
                for budget in previous_attempt_budgets
                if isinstance(budget, dict)
            ]
            if isinstance(previous_attempt_budgets, list)
            else []
        )
        previous_budget = previous_usage.get("budget")
        if isinstance(previous_budget, dict):
            attempt_budgets.append(dict(previous_budget))
        if attempt_budgets:
            payload["attempt_budgets"] = attempt_budgets

        counters = dict(payload.get("counters") or {})
        for name, raw_count in previous_counters.items():
            if name == "provider_cursor_open":
                # This is a last-attempt work-availability gauge, not lifetime
                # telemetry. Carrying an old open cursor past the retry ceiling
                # would create an unbounded linked recovery job.
                continue
            try:
                prior_count = max(0, int(raw_count or 0))
                current_count = max(0, int(counters.get(name) or 0))
            except (TypeError, ValueError, OverflowError):
                continue
            counters[name] = prior_count + current_count
        counters["analyzed_sources"] = len(completed_source_ids)
        counters["durable_attempts"] = max(
            1,
            int(job_row.get("attempt_count") or 1),
        )
        payload["counters"] = counters
        payload["consumed_video_ids"] = sorted(
            video_id
            for raw_id in completed_source_ids
            if (video_id := normalize_youtube_video_id(raw_id)) is not None
        )
        completed_ids = set(payload["consumed_video_ids"])
        capacity_deferred_ids = sorted(
            video_id
            for raw_id in capacity_deferred_source_ids
            if (
                (video_id := normalize_youtube_video_id(raw_id)) is not None
                and video_id not in completed_ids
            )
        )
        payload["capacity_deferred_video_ids"] = capacity_deferred_ids
        capacity_deferred_id_set = set(capacity_deferred_ids)
        failed_ids = sorted(
            video_id
            for raw_id in attempted_source_ids
            if (
                (video_id := normalize_youtube_video_id(raw_id)) is not None
                and video_id not in completed_ids
                and video_id not in capacity_deferred_id_set
            )
        )
        failed_source_attempts = dict(previous_failed_source_attempts)
        for video_id in failed_ids:
            # A known provider failure is a real source-analysis attempt even
            # when the durable job itself remains active. Unknown crash/lease
            # recovery does not enter this path and therefore is not invented.
            failed_source_attempts[video_id] = min(
                100,
                failed_source_attempts.get(video_id, 0) + 1,
            )
        for video_id in completed_ids | capacity_deferred_id_set:
            failed_source_attempts.pop(video_id, None)
        payload["failed_source_attempts"] = dict(sorted(failed_source_attempts.items()))
        retry_errors = list(previous_retry_errors)
        if retry_error is not None:
            retry_errors.append(dict(retry_error))
        if retry_errors:
            payload["retry_errors"] = retry_errors
        if terminal:
            summary = dict(payload.get("summary") or {})
            budget = payload.get("budget")
            gemini_budget = (
                budget.get("gemini")
                if isinstance(budget, dict)
                and isinstance(budget.get("gemini"), dict)
                else {}
            )
            record_known_cost = max(
                0.0,
                float(summary.get("known_billed_cost_usd") or 0.0),
                float(summary.get("estimated_cost_usd") or 0.0),
                float(summary.get("telemetry_priced_cost_usd") or 0.0),
            )
            record_unknown_cost = max(
                0.0,
                float(
                    summary.get("billing_unknown_reserved_cost_usd")
                    or 0.0
                ),
            )
            groq_known_cost = max(
                0.0,
                float(summary.get("groq_known_billed_cost_usd") or 0.0),
            )
            groq_unknown_cost = max(
                0.0,
                float(
                    summary.get(
                        "groq_billing_unknown_reserved_cost_usd"
                    )
                    or 0.0
                ),
            )
            record_gemini_known_cost = max(
                0.0,
                record_known_cost - groq_known_cost,
            )
            record_gemini_unknown_cost = max(
                0.0,
                record_unknown_cost - groq_unknown_cost,
            )
            current_exposure = max(
                0.0,
                float(gemini_budget.get("cost_exposure_usd") or 0.0),
            )
            budget_committed_cost = max(
                0.0,
                float(gemini_budget.get("committed_cost_usd") or 0.0),
            )
            budget_unknown_cost = min(
                budget_committed_cost,
                max(
                    0.0,
                    float(
                        gemini_budget.get(
                            "billing_unknown_cost_exposure_usd"
                        )
                        or 0.0
                    ),
                ),
            )
            budget_known_cost = max(
                0.0,
                budget_committed_cost - budget_unknown_cost,
            )
            known_cost = (
                max(record_gemini_known_cost, budget_known_cost)
                + groq_known_cost
            )
            missing_known_cost = max(
                0.0,
                budget_known_cost - record_gemini_known_cost,
            )
            unknown_cost = (
                max(record_gemini_unknown_cost, budget_unknown_cost)
                + groq_unknown_cost
            )
            residual_unknown_cost = max(
                0.0,
                current_exposure
                - (known_cost - groq_known_cost)
                - (unknown_cost - groq_unknown_cost),
            )
            unknown_cost += residual_unknown_cost
            for field in (
                "estimated_cost_usd",
                "known_billed_cost_usd",
                "telemetry_priced_cost_usd",
            ):
                summary[field] = round(
                    max(float(summary.get(field) or 0.0), known_cost),
                    8,
                )
            if missing_known_cost > 1e-9:
                summary["unattributed_known_billed_cost_usd"] = round(
                    missing_known_cost,
                    8,
                )
            else:
                summary.pop("unattributed_known_billed_cost_usd", None)
            missing_unknown_cost = max(
                0.0,
                budget_unknown_cost - record_gemini_unknown_cost,
            ) + residual_unknown_cost
            summary["billing_unknown_reserved_cost_usd"] = round(
                unknown_cost,
                8,
            )
            summary["current_cost_exposure_usd"] = round(
                known_cost + unknown_cost,
                8,
            )
            if missing_unknown_cost > 1e-9:
                summary["unattributed_billing_unknown_cost_usd"] = round(
                    missing_unknown_cost,
                    8,
                )
            else:
                summary.pop("unattributed_billing_unknown_cost_usd", None)
            accepted_clips = max(
                0,
                int(summary.get("accepted_clips") or 0),
            )
            if unknown_cost > 1e-9 or not accepted_clips:
                summary["cost_per_accepted_clip_usd"] = None
            else:
                summary["cost_per_accepted_clip_usd"] = round(
                    known_cost / accepted_clips,
                    8,
                )
            payload["summary"] = summary
        return payload

    def cancel_if_adaptation_stale(conn) -> bool:
        stored_schema = str(params.get("request_schema_version") or "")
        stored_fingerprint = str(params.get("adaptation_fingerprint") or "")
        stored_knowledge_level = str(
            params.get("knowledge_level") or "beginner"
        )
        current_knowledge_level = str(
            reel_service.learner_progress(conn, material_id, learner_id).get(
                "selected_level"
            )
            or "beginner"
        )
        is_current = (
            stored_schema == GENERATION_REQUEST_SCHEMA_VERSION
            and stored_knowledge_level == current_knowledge_level
            and stored_fingerprint == _learner_adaptation_fingerprint(
                conn,
                material_id=material_id,
                learner_id=learner_id,
            )
        )
        if is_current:
            return False
        transition_generation_terminal(
            conn,
            job_id=job_id,
            status="cancelled",
            result_generation_id=generation_id or None,
            lease_owner=lease_owner,
            usage=generation_usage_payload(terminal=True),
        )
        logger.info("cancelled stale adaptive generation job_id=%s", job_id)
        return True

    def checkpoint_yielded_usage() -> bool:
        usage = generation_usage_payload(terminal=True)
        try:
            checkpointed = bool(
                _run_generation_db_transaction(
                    "yielded_attempt_usage_checkpoint",
                    lambda checkpoint_conn: checkpoint_generation_yielded_usage(
                        checkpoint_conn,
                        job_id=job_id,
                        lease_owner=lease_owner,
                        expected_attempt_count=durable_attempt_count,
                        usage=usage,
                    ),
                    replay_after_unknown_commit=True,
                )
            )
        except Exception:
            logger.exception(
                "generation yielded-usage checkpoint failed job_id=%s",
                job_id,
            )
            return False
        logger.info(
            "generation worker yielded lease job_id=%s usage_checkpointed=%s",
            job_id,
            checkpointed,
        )
        return checkpointed

    heartbeat_thread.start()
    try:
        def setup_generation(setup_conn: Any) -> str | None:
            if not update_generation_progress(
                setup_conn,
                job_id=job_id,
                lease_owner=lease_owner,
                phase="retrieval",
                progress=0.05,
            ):
                raise JobLeaseLostError(
                    f"generation job lease is no longer active: {job_id}"
                )
            if cancel_if_adaptation_stale(setup_conn):
                return None
            setup_generation_id = setup_generation_candidate_id
            if not _fetch_generation_row(setup_conn, setup_generation_id):
                source_generation_id = (
                    str(job_row.get("source_generation_id") or "").strip() or None
                )
                setup_generation_id = _create_generation_row(
                    setup_conn,
                    material_id=material_id,
                    concept_id=concept_id,
                    request_key=str(job_row.get("request_key") or ""),
                    generation_mode=mode,
                    retrieval_profile="unified",
                    source_generation_id=source_generation_id,
                    generation_id=setup_generation_id,
                )
            attached_at = now_iso()
            attached = execute_modify(
                setup_conn,
                "UPDATE reel_generation_jobs SET result_generation_id = ?, updated_at = ? "
                "WHERE id = ? AND status = 'running' AND lease_owner = ? "
                "AND cancel_requested = 0 AND lease_expires_at > ? "
                "AND (deadline_at IS NULL OR deadline_at > ?) "
                "AND (result_generation_id IS NULL OR TRIM(result_generation_id) = '' "
                "OR result_generation_id = ?)",
                (
                    setup_generation_id,
                    attached_at,
                    job_id,
                    lease_owner,
                    attached_at,
                    attached_at,
                    setup_generation_id,
                ),
            )
            if not attached:
                raise JobLeaseLostError(
                    f"generation job lease is no longer active: {job_id}"
                )
            return setup_generation_id

        setup_generation_id = _run_generation_db_transaction(
            "setup",
            setup_generation,
            retry_should_stop=db_retry_should_stop,
            replay_after_unknown_commit=True,
        )
        if setup_generation_id is None:
            return
        generation_id = setup_generation_id

        with get_conn() as conn:
            source_generation_id = (
                str(job_row.get("source_generation_id") or "").strip() or None
            )
            generation_has_lesson_order = (
                _stored_generation_lesson_order_ids(conn, generation_id) is not None
            )
            source_generation_rows: dict[str, dict[str, Any]] = {}
            source_generation_ids = _response_generation_ids(
                conn,
                source_generation_id,
                generation_rows_out=source_generation_rows,
            )
            prior_reacquisition_checkpoints: list[bool] = []
            (
                same_adaptation_restatement_ids,
                same_adaptation_restatement_guard_ids,
            ) = _same_adaptation_current_restatement_policy(
                source_generation_rows,
                source_generation_ids,
                adaptation_fingerprint=params.get(
                    "adaptation_fingerprint"
                ),
                reacquisition_checkpoint_out=(
                    prior_reacquisition_checkpoints
                ),
            )
            prior_reacquisition_checkpoint_seen = any(
                prior_reacquisition_checkpoints
            )
            prior_consumed_video_ids = _generation_chain_consumed_video_ids(
                conn,
                generation_id=source_generation_id or "",
            )
            prior_consumed_video_ids.update(previous_consumed_source_ids)
            prior_failed_source_attempts = _generation_chain_failed_source_attempts(
                conn,
                generation_id=source_generation_id or "",
            )
            for video_id, count in previous_failed_source_attempts.items():
                prior_failed_source_attempts[video_id] = min(
                    100,
                    prior_failed_source_attempts.get(video_id, 0) + count,
                )
            retired_failed_video_ids = {
                video_id
                for video_id, count in prior_failed_source_attempts.items()
                if count >= SOURCE_ANALYSIS_MAX_ATTEMPTS
            }
            source_reel_count = _current_level_reusable_generation_reel_count(
                conn,
                generation_id=source_generation_id,
                material_id=material_id,
                concept_id=concept_id,
                learner_id=learner_id,
                request_params=params,
                requested=requested_count,
                editorial_excluded_reel_ids=(
                    same_adaptation_restatement_ids
                ),
                editorial_guard_reel_ids=(
                    same_adaptation_restatement_guard_ids
                ),
            )
            analyzed_source_budget = _generation_chain_analyzed_source_budget(
                conn,
                generation_id=source_generation_id or "",
            )
            if is_continuation or has_fresh_source_budget:
                # A continuation gets a fresh provider budget, but verified
                # source-chain clips that were never delivered fill the batch
                # before another paid search. An initial cross-request cache
                # top-up also gets one fresh bounded budget when that cache is
                # strict but too small for the nine-reel startup reservoir.
                analyzed_source_budget = 0
            analyzed_source_budget += len(previous_consumed_source_ids)
            remaining_source_budget = max(
                0,
                GENERATION_SOURCE_BUDGETS[mode] - analyzed_source_budget,
            )
            if is_continuation and source_reel_count > 0:
                # Return an unseen cached sibling immediately, even when it is
                # smaller than the requested batch. The following continuation
                # receives the fresh provider budget after cached inventory is
                # drained, so playback never waits on Gemini after clip three.
                remaining_source_budget = 0
            emitted_reel_ids: set[str] = set()
            emitted_reel_order: list[str] = []
            emitted_clip_keys: set[str] = set()
            emitted_count = 0
            if durable_attempt_count > 1:
                prior_candidate_reels = [
                    payload.get("reel")
                    for event in replay_generation_events(conn, job_id=job_id)
                    if str(event.get("type") or "") == "candidate"
                    and isinstance((payload := event.get("payload")), dict)
                    and payload.get("provisional") is True
                    and isinstance(payload.get("reel"), dict)
                ]
                usable_prior_ids = _usable_boundary_reel_ids(
                    conn,
                    [
                        str(reel.get("reel_id") or "")
                        for reel in prior_candidate_reels
                    ],
                )
                for reel in prior_candidate_reels:
                    reel_id, clip_key = _reel_identity_key(reel)
                    if not reel_id or reel_id not in usable_prior_ids:
                        continue
                    meaningful_clip_key = (
                        clip_key if _reel_source_video_id(reel) else ""
                    )
                    if (
                        reel_id in emitted_reel_ids
                        or (
                            meaningful_clip_key
                            and meaningful_clip_key in emitted_clip_keys
                        )
                    ):
                        continue
                    emitted_reel_ids.add(reel_id)
                    emitted_reel_order.append(reel_id)
                    if meaningful_clip_key:
                        emitted_clip_keys.add(meaningful_clip_key)
                    emitted_count += 1

            def on_candidate(reel: dict[str, Any]) -> None:
                nonlocal emitted_count
                if should_cancel():
                    raise GenerationCancelledError("Generation cancelled.")
                reel_id, clip_key = _reel_identity_key(reel)
                meaningful_clip_key = (
                    clip_key if _reel_source_video_id(reel) else ""
                )
                if (
                    not reel_id
                    or reel_id in emitted_reel_ids
                    or (
                        meaningful_clip_key
                        and meaningful_clip_key in emitted_clip_keys
                    )
                    or emitted_count >= requested_count
                ):
                    return
                candidate_payload = {
                    "reel": _public_generation_reel(reel),
                    "provisional": True,
                }
                try:
                    append_generation_event(
                        conn,
                        job_id=job_id,
                        event_type="candidate",
                        payload=candidate_payload,
                        lease_owner=lease_owner,
                    )
                except Exception:
                    def append_candidate_if_absent(event_conn: Any) -> Any:
                        for existing_event in replay_generation_events(
                            event_conn,
                            job_id=job_id,
                            limit=2000,
                        ):
                            if str(existing_event.get("type") or "") != "candidate":
                                continue
                            existing_payload = existing_event.get("payload")
                            if (
                                not isinstance(existing_payload, dict)
                                or existing_payload.get("provisional") is not True
                                or not isinstance(existing_payload.get("reel"), dict)
                            ):
                                continue
                            existing_reel = existing_payload["reel"]
                            existing_reel_id, existing_clip_key = (
                                _reel_identity_key(existing_reel)
                            )
                            if existing_reel_id == reel_id or (
                                meaningful_clip_key
                                and _reel_source_video_id(existing_reel)
                                and existing_clip_key == meaningful_clip_key
                            ):
                                return existing_event
                        return append_generation_event(
                            event_conn,
                            job_id=job_id,
                            event_type="candidate",
                            payload=candidate_payload,
                            lease_owner=lease_owner,
                        )

                    _run_generation_db_transaction(
                        "candidate_event",
                        append_candidate_if_absent,
                        retry_should_stop=db_retry_should_stop,
                        replay_after_unknown_commit=True,
                    )
                emitted_reel_ids.add(reel_id)
                emitted_reel_order.append(reel_id)
                if meaningful_clip_key:
                    emitted_clip_keys.add(meaningful_clip_key)
                emitted_count += 1

            base_exclusions = list(params.get("exclude_video_ids") or [])
            base_exclusions.extend(sorted(retired_failed_video_ids))
            prior_covered_intent_obligation_keys: set[str] | None = None

            def run_retrieval_stage(
                *,
                retrieval_profile: Literal["deep"],
                video_budget: int,
                new_reel_cap: int,
                excluded_video_ids: list[str],
                consumed_video_ids: set[str],
                analyzed_video_ids: set[str],
                attempted_video_ids: set[str],
                capacity_deferred_video_ids: set[str],
                retrieved_video_ids: set[str],
            ) -> None:
                nonlocal prior_covered_intent_obligation_keys
                if prior_covered_intent_obligation_keys is None:
                    prior_release_ids = (
                        _authoritative_release_reel_ids(
                            conn,
                            source_generation_id,
                        )
                        or []
                    )
                    prior_coverage, _recent_prior_coverage = (
                        _lesson_prior_coverage(
                            conn,
                            material_id=material_id,
                            reel_ids=prior_release_ids,
                        )
                    )
                    prior_covered_intent_obligation_keys = set().union(*(
                        {
                            str(value).strip()
                            for value in (
                                item.get("intent_obligation_keys") or ()
                            )
                            if str(value or "").strip()
                        }
                        for item in prior_coverage
                    )) if prior_coverage else set()
                reel_service.generate_reels(
                    conn,
                    material_id=material_id,
                    concept_id=concept_id,
                    num_reels=requested_count,
                    creative_commons_only=bool(params.get("creative_commons_only")),
                    exclude_video_ids=excluded_video_ids,
                    consumed_video_ids=sorted(consumed_video_ids),
                    exclude_generation_ids=source_generation_ids,
                    fast_mode=mode == "fast",
                    preferred_video_duration=_normalize_preferred_video_duration(
                        str(params.get("preferred_video_duration") or "any")
                    ),
                    target_clip_duration_sec=0,
                    target_clip_duration_min_sec=None,
                    target_clip_duration_max_sec=None,
                    retrieval_profile=retrieval_profile,
                    generation_id=generation_id,
                    min_relevance_threshold=float(params.get("min_relevance") or 0.0),
                    page_hint=1,
                    on_reel_created=on_candidate,
                    should_cancel=should_cancel,
                    knowledge_level_override=str(params.get("knowledge_level") or "beginner"),
                    learner_id=learner_id,
                    generation_context=context,
                    max_generation_videos=video_budget,
                    acquisition_concept_offset=0,
                    max_new_reels=new_reel_cap,
                    analyzed_video_ids=analyzed_video_ids,
                    attempted_video_ids=attempted_video_ids,
                    capacity_deferred_video_ids=capacity_deferred_video_ids,
                    retrieved_video_ids=retrieved_video_ids,
                    covered_intent_obligation_keys=(
                        prior_covered_intent_obligation_keys
                    ),
                )

            current_count = source_reel_count + _count_generation_surfaceable_reels(
                conn, generation_id
            )
            retrieval_provider_error: ClipEngineProviderError | None = None
            if (
                not generation_has_lesson_order
                and current_count < requested_count
                and remaining_source_budget > 0
            ):
                context.budget.reserve_pass()
                if should_cancel():
                    raise GenerationCancelledError("Generation cancelled.")
                try:
                    run_retrieval_stage(
                        retrieval_profile="deep",
                        video_budget=remaining_source_budget,
                        new_reel_cap=requested_count - current_count,
                        excluded_video_ids=base_exclusions,
                        consumed_video_ids=prior_consumed_video_ids,
                        analyzed_video_ids=completed_source_ids,
                        attempted_video_ids=attempted_source_ids,
                        capacity_deferred_video_ids=capacity_deferred_source_ids,
                        retrieved_video_ids=retrieved_video_ids,
                    )
                except ClipEngineProviderError as exc:
                    current_count = (
                        source_reel_count
                        + _count_generation_surfaceable_reels(conn, generation_id)
                    )
                    fallback_job = {
                        **(get_generation_job(conn, job_id) or job_row),
                        "result_generation_id": generation_id,
                    }
                    fallback_prior_unseen: list[dict[str, Any]] = []
                    fallback_reels = _generation_job_reels(
                        conn,
                        fallback_job,
                        organizer_candidate_limit=lesson_candidate_limit,
                        apply_release_order=False,
                        preserve_lesson_order_metadata=True,
                        prior_unseen_reels_out=(
                            fallback_prior_unseen
                            if source_generation_id
                            and _stored_generation_lesson_order_ids(
                                conn,
                                generation_id,
                            )
                            is None
                            else None
                        ),
                        editorial_excluded_reel_ids=(
                            same_adaptation_restatement_ids
                        ),
                        editorial_guard_reel_ids=(
                            same_adaptation_restatement_guard_ids
                        ),
                    )
                    reusable_inventory_count = len(
                        fallback_reels
                    ) + len(fallback_prior_unseen)
                    if reusable_inventory_count <= 0:
                        raise
                    retrieval_provider_error = exc
                    logger.warning(
                        "fresh retrieval failed; reusing validated inventory "
                        "job_id=%s inventory_count=%d error=%s",
                        job_id,
                        reusable_inventory_count,
                        exc.as_dict(),
                    )
                current_count = (
                    source_reel_count
                    + _count_generation_surfaceable_reels(conn, generation_id)
                )
                if not update_generation_progress(
                    conn,
                    job_id=job_id,
                    lease_owner=lease_owner,
                    phase="ranking",
                    progress=0.85,
                    usage=generation_usage_payload(),
                ):
                    raise JobLeaseLostError(
                        f"generation job lease is no longer active: {job_id}"
                    )

            cumulative_count = (
                source_reel_count
                + _count_generation_surfaceable_reels(conn, generation_id)
            )
            if cancel_if_adaptation_stale(conn):
                return
            has_verified_reservoir = (
                False
                if is_continuation
                else _verified_reusable_generation_chain(
                    conn,
                    generation_id=generation_id,
                    material_id=material_id,
                )
            )
            refreshed_job = get_generation_job(conn, job_id) or {
                **job_row,
                "result_generation_id": generation_id,
            }
            stored_order_ids = _stored_generation_lesson_order_ids(
                conn,
                generation_id,
            )
            prior_unseen_reels: list[dict[str, Any]] = []
            collect_prior_unseen = bool(
                source_generation_id and stored_order_ids is None
            )
            rankable_fallback = (
                []
                if cumulative_count or has_verified_reservoir
                else _generation_job_reels(
                    conn,
                    refreshed_job,
                    organizer_candidate_limit=lesson_candidate_limit,
                    apply_release_order=False,
                    preserve_lesson_order_metadata=True,
                    prior_unseen_reels_out=(
                        prior_unseen_reels if collect_prior_unseen else None
                    ),
                    editorial_excluded_reel_ids=same_adaptation_restatement_ids,
                    editorial_guard_reel_ids=(
                        same_adaptation_restatement_guard_ids
                    ),
                )
            )
            stage_counters = context.counters()
            stage_counters["analyzed_sources"] = len(completed_source_ids)
            provider_cursor_open = int(
                stage_counters.get("provider_cursor_open") or 0
            ) > 0
            ordering_degraded = False
            recall_preparation: dict[str, int] | None = None
            activate_generation = False
            reconciliation_tail_reel_ids: list[str] | None = None
            locked_prefix_reel_ids: list[str] = []
            candidate_final_reels: list[dict[str, Any]] = []
            authoritative_prior_reel_ids: list[str] = []
            editorial_reacquisition_available = False
            if cumulative_count or has_verified_reservoir or rankable_fallback:
                candidate_final_reels = rankable_fallback or _generation_job_reels(
                    conn,
                    refreshed_job,
                    organizer_candidate_limit=lesson_candidate_limit,
                    preserve_lesson_order_metadata=True,
                    prior_unseen_reels_out=(
                        prior_unseen_reels if collect_prior_unseen else None
                    ),
                    editorial_excluded_reel_ids=same_adaptation_restatement_ids,
                    editorial_guard_reel_ids=(
                        same_adaptation_restatement_guard_ids
                    ),
                )
                if stored_order_ids is None:
                    candidate_final_reels = (
                        _filter_continuation_release_temporal_overlaps(
                            conn,
                            source_generation_id=source_generation_id,
                            generation_id=generation_id,
                            reels=candidate_final_reels,
                            prior_reel_ids_out=authoritative_prior_reel_ids,
                        )
                    )
            if candidate_final_reels or prior_unseen_reels:
                final_reels = candidate_final_reels
                if stored_order_ids is None:
                    surfaceable_candidate_ids = {
                        str(reel.get("reel_id") or "")
                        for reel in final_reels
                        if str(reel.get("reel_id") or "")
                    }
                    locked_prefix_reel_ids = (
                        _live_candidate_locked_prefix_ids(
                            conn,
                            material_id=material_id,
                            learner_id=learner_id,
                            candidate_reel_ids=[
                                reel_id
                                for reel_id in emitted_reel_order
                                if reel_id in surfaceable_candidate_ids
                            ],
                            job_started_at=refreshed_job.get("started_at"),
                        )
                    )
                    remediation_concept_ids: list[str] = []
                    concept_signals = _learner_concept_signals(
                        conn,
                        material_id=material_id,
                        learner_id=learner_id,
                        propagate_concept_families=True,
                        candidate_concept_ids={
                            str(reel.get("concept_id") or "")
                            for reel in (
                                candidate_final_reels + prior_unseen_reels
                            )
                            if str(reel.get("concept_id") or "")
                        },
                        remediation_concept_ids_out=remediation_concept_ids,
                    )
                    prior_reel_ids = (
                        authoritative_prior_reel_ids
                        or _continuation_delivered_reel_ids(
                            conn,
                            str(params.get("continuation_token") or "").strip(),
                        )
                    )
                    prior_reel_rank = {
                        reel_id: index
                        for index, reel_id in enumerate(prior_reel_ids)
                    }
                    prior_unseen_by_id = {
                        reel_id: reel
                        for reel in prior_unseen_reels
                        if (
                            reel_id := str(reel.get("reel_id") or "").strip()
                        ) in prior_reel_rank
                        and reel_id not in surfaceable_candidate_ids
                    }
                    ordered_prior_unseen_reels = [
                        prior_unseen_by_id[reel_id]
                        for reel_id in prior_reel_ids
                        if reel_id in prior_unseen_by_id
                    ]
                    required_prior_unseen_ids = [
                        str(reel.get("reel_id") or "")
                        for reel in ordered_prior_unseen_reels
                    ]
                    required_organizer_reel_ids = list(dict.fromkeys([
                        *locked_prefix_reel_ids,
                        *required_prior_unseen_ids,
                    ]))
                    prior_history_ids = (
                        [
                            reel_id
                            for reel_id in prior_reel_ids
                            if reel_id not in prior_unseen_by_id
                        ]
                        if is_continuation
                        else prior_reel_ids
                    )
                    prior_history_ids = list(dict.fromkeys([
                        *prior_history_ids,
                        *_learner_signal_reel_ids(
                            conn,
                            material_id=material_id,
                            learner_id=learner_id,
                        ),
                    ]))
                    (
                        prior_concept_coverage,
                        recent_prior_objective_coverage,
                    ) = _lesson_prior_coverage(
                        conn,
                        material_id=material_id,
                        reel_ids=prior_history_ids,
                    )
                    organizer_candidates = [
                        *ordered_prior_unseen_reels,
                        *final_reels,
                    ]
                    ordering = order_lesson_batch(
                        organizer_candidates,
                        topic=_lesson_order_topic(
                            conn,
                            material_id=material_id,
                            reels=organizer_candidates,
                        ),
                        learner_level=str(params.get("knowledge_level") or "beginner"),
                        learner_difficulty_target=params.get(
                            "effective_level_target"
                        ),
                        concept_signals=concept_signals,
                        remediation_concept_ids=remediation_concept_ids,
                        # The request count controls acquisition/stream cadence,
                        # not Gemini's editorial subset. Every already-verified
                        # candidate in this bounded pool may be included or
                        # omitted without adding provider or boundary work.
                        release_limit=len(organizer_candidates),
                        required_reel_ids=required_organizer_reel_ids,
                        locked_prefix_reel_ids=locked_prefix_reel_ids,
                        prior_concept_coverage=prior_concept_coverage,
                        recent_prior_objective_coverage=(
                            recent_prior_objective_coverage
                        ),
                        should_cancel=should_cancel,
                        generation_context=context,
                    )
                    final_reels = [
                        reel
                        for reel in ordering.reels
                        if str(reel.get("reel_id") or "")
                        in surfaceable_candidate_ids
                    ]
                    selected_current_by_id = {
                        str(reel.get("reel_id") or ""): reel
                        for reel in final_reels
                    }
                    selected_current_ids = [
                        str(reel.get("reel_id") or "")
                        for reel in final_reels
                    ]
                    released_ordered_ids = [
                        reel_id
                        for reel_id in locked_prefix_reel_ids
                        if reel_id in selected_current_by_id
                    ] + [
                        reel_id
                        for reel_id in selected_current_ids
                        if reel_id not in locked_prefix_reel_ids
                    ]
                    final_reels = [
                        selected_current_by_id[reel_id]
                        for reel_id in released_ordered_ids
                    ]
                    organizer_ordered_ids = getattr(
                        ordering,
                        "ordered_reel_ids",
                        None,
                    )
                    if not isinstance(organizer_ordered_ids, list):
                        organizer_ordered_ids = [
                            str(reel.get("reel_id") or "")
                            for reel in ordering.reels
                        ]
                    organizer_ordered_ids = [
                        reel_id
                        for reel_id in locked_prefix_reel_ids
                        if reel_id in organizer_ordered_ids
                    ] + [
                        reel_id
                        for reel_id in organizer_ordered_ids
                        if reel_id not in locked_prefix_reel_ids
                    ]
                    reconciliation_tail_reel_ids = (
                        organizer_ordered_ids
                        if required_prior_unseen_ids
                        and set(required_prior_unseen_ids).issubset(
                            organizer_ordered_ids
                        )
                        else None
                    )
                    terminal_summary_start_reel_id = (
                        _surviving_terminal_summary_start_reel_id(
                            ordered_reel_ids=organizer_ordered_ids,
                            terminal_summary_start_reel_id=getattr(
                                ordering,
                                "terminal_summary_start_reel_id",
                                None,
                            ),
                            surviving_reel_ids=released_ordered_ids,
                        )
                    )
                    released_id_set = set(released_ordered_ids)
                    released_checkpoint_ids = (
                        [
                            reel_id
                            for reel_id in ordering.assessment_checkpoint_reel_ids
                            if reel_id in released_id_set
                        ]
                        if ordering.assessment_checkpoint_reel_ids is not None
                        else None
                    )
                    organizer_prior_restatement_ids = getattr(
                        ordering,
                        "prior_restatement_reel_ids",
                        None,
                    )
                    if not isinstance(organizer_prior_restatement_ids, list):
                        organizer_prior_restatement_ids = []
                    organizer_current_restatement_ids = getattr(
                        ordering,
                        "current_restatement_reel_ids",
                        None,
                    )
                    if not isinstance(organizer_current_restatement_ids, list):
                        organizer_current_restatement_ids = []
                    editorial_fallback_reason: str | None = None
                    current_candidate_reels = [
                        reel
                        for reel in candidate_final_reels
                        if str(
                            reel.get("reel_id") or ""
                        ).strip()
                    ]
                    current_candidate_ids = [
                        str(reel.get("reel_id") or "").strip()
                        for reel in current_candidate_reels
                    ]
                    current_restatement_id_set = set(
                        organizer_current_restatement_ids
                    )
                    adaptation_fingerprint = str(
                        params.get("adaptation_fingerprint") or ""
                    ).strip()
                    restatement_only_current_omission = bool(
                        is_continuation
                        and adaptation_fingerprint
                        and organizer_ordered_ids
                        and not released_ordered_ids
                        and current_candidate_ids
                        and set(current_candidate_ids).issubset(
                            current_restatement_id_set
                        )
                        and current_restatement_id_set.isdisjoint(
                            organizer_ordered_ids
                        )
                    )
                    if (
                        restatement_only_current_omission
                        and prior_reacquisition_checkpoint_seen
                    ):
                        # One empty editorial checkpoint already bought a fresh
                        # bounded search. If every new valid candidate is again
                        # only a restatement, surface the strongest current row
                        # instead of creating an unbounded chain of empty partials.
                        fallback_reel_id = current_candidate_ids[0]
                        final_reels = [current_candidate_reels[0]]
                        released_ordered_ids = [fallback_reel_id]
                        organizer_current_restatement_ids = [
                            reel_id
                            for reel_id in organizer_current_restatement_ids
                            if reel_id != fallback_reel_id
                        ]
                        editorial_fallback_reason = (
                            "restatement_reacquisition_limit"
                        )
                        if reconciliation_tail_reel_ids is not None:
                            reconciliation_tail_reel_ids = list(
                                dict.fromkeys([
                                    *reconciliation_tail_reel_ids,
                                    fallback_reel_id,
                                ])
                            )
                    ordering_degraded = bool(
                        ordering.degraded or editorial_fallback_reason
                    )
                    checkpoint_ids = assessment_checkpoint_reel_ids(
                        released_ordered_ids,
                        released_checkpoint_ids,
                        degraded=ordering_degraded,
                    )
                    lesson_order_metadata = {
                        "version": 2,
                        "prompt_version": LESSON_ORDER_PROMPT_VERSION,
                        "ordered_reel_ids": released_ordered_ids,
                        "reconciliation_tail_reel_ids": (
                            reconciliation_tail_reel_ids
                        ),
                        "locked_prefix_reel_ids": locked_prefix_reel_ids,
                        "assessment_checkpoint_reel_ids": checkpoint_ids,
                        "prior_restatement_reel_ids": (
                            organizer_prior_restatement_ids
                        ),
                        "current_restatement_reel_ids": (
                            organizer_current_restatement_ids
                        ),
                        "current_restatement_guard_reel_ids": (
                            organizer_ordered_ids
                            if organizer_current_restatement_ids
                            else []
                        ),
                        "editorial_reacquisition_checkpoint": False,
                        "adaptation_fingerprint": str(
                            params.get("adaptation_fingerprint") or ""
                        ).strip(),
                        "terminal_summary_start_reel_id": (
                            terminal_summary_start_reel_id
                        ),
                        "model_used": ordering.model_used,
                        "created_at": now_iso(),
                        "degraded": ordering_degraded,
                        "fallback_reason": (
                            editorial_fallback_reason
                            or ordering.fallback_reason
                        ),
                        "provider_called": ordering.provider_called,
                    }
                    editorial_reacquisition_available = bool(
                        restatement_only_current_omission
                        and not prior_reacquisition_checkpoint_seen
                    )
                    lesson_order_metadata[
                        "editorial_reacquisition_checkpoint"
                    ] = editorial_reacquisition_available
                    _run_generation_db_transaction(
                        "lesson_order_metadata",
                        lambda lesson_conn: _persist_generation_lesson_order(
                            lesson_conn,
                            generation_id=generation_id,
                            metadata=lesson_order_metadata,
                        ),
                        retry_should_stop=db_retry_should_stop,
                    )
                else:
                    stored_metadata = (
                        _stored_generation_lesson_order_metadata(conn, generation_id)
                        or {}
                    )
                    ordering_degraded = (
                        bool(stored_metadata.get("degraded")) or not stored_order_ids
                    )
                    final_reels = _apply_generation_lesson_order(
                        conn,
                        generation_id=generation_id,
                        reels=final_reels,
                    )
                    final_reels = _filter_continuation_release_temporal_overlaps(
                        conn,
                        source_generation_id=source_generation_id,
                        generation_id=generation_id,
                        reels=final_reels,
                    )
                final_reels = [_public_generation_reel(reel) for reel in final_reels]

                lesson_order_metadata = (
                    _stored_generation_lesson_order_metadata(conn, generation_id) or {}
                )
                reconciliation_tail_reel_ids = (
                    _lesson_reconciliation_tail_ids(lesson_order_metadata)
                )
                ordered_reel_ids = lesson_order_metadata.get("ordered_reel_ids")
                organizer_checkpoint_ids = (
                    assessment_checkpoint_reel_ids(
                        ordered_reel_ids,
                        lesson_order_metadata.get("assessment_checkpoint_reel_ids"),
                        degraded=lesson_order_metadata.get("degraded") is not False,
                    )
                    if lesson_order_metadata.get("version") == 2
                    and isinstance(ordered_reel_ids, list)
                    else None
                )
                final_reel_id_set = {
                    str(reel.get("reel_id") or "") for reel in final_reels
                }
                checkpoint_reel_ids = [
                    reel_id
                    for reel_id in organizer_checkpoint_ids or []
                    if reel_id in final_reel_id_set
                ]
                if checkpoint_reel_ids:
                    try:
                        recall_preparation = _run_generation_db_transaction(
                            "recall_preparation",
                            lambda recall_conn: assessment_service.prepare_reel_questions(
                                recall_conn,
                                reel_ids=checkpoint_reel_ids,
                                should_cancel=should_cancel,
                                use_model=False,
                            ),
                            retry_should_stop=db_retry_should_stop,
                        )
                    except AssessmentCancelledError as exc:
                        raise GenerationCancelledError(str(exc)) from exc
                    except Exception:
                        logger.exception(
                            "organizer checkpoint preparation failed before release "
                            "job_id=%s",
                            job_id,
                        )
                        recall_preparation = None
                    recall_preparation_complete = bool(
                        recall_preparation
                        and recall_preparation["requested"] == len(checkpoint_reel_ids)
                        and recall_preparation["prepared"] == len(checkpoint_reel_ids)
                    )
                    if not recall_preparation_complete:
                        degraded_metadata = dict(lesson_order_metadata)
                        degraded_metadata.update(
                            {
                                "assessment_checkpoint_reel_ids": None,
                                "degraded": True,
                                "fallback_reason": "recall_preparation_unavailable",
                                "recall_available": False,
                            }
                        )
                        _run_generation_db_transaction(
                            "lesson_order_recall_degradation",
                            lambda lesson_conn: _persist_generation_lesson_order(
                                lesson_conn,
                                generation_id=generation_id,
                                metadata=degraded_metadata,
                            ),
                            retry_should_stop=db_retry_should_stop,
                        )
                        ordering_degraded = True
                        logger.warning(
                            "recall unavailable; releasing clips without automatic "
                            "checkpoints job_id=%s requested=%d prepared=%d",
                            job_id,
                            len(checkpoint_reel_ids),
                            int((recall_preparation or {}).get("prepared") or 0),
                        )
                activate_generation = True
            elif has_verified_reservoir:
                # Preserve a verified reservoir when another operational guard
                # leaves no releasable row; learner difficulty is not such a guard.
                final_reels = []
                activate_generation = True
            elif provider_cursor_open:
                _complete_generation(
                    conn,
                    generation_id=generation_id,
                    retrieval_profile="unified",
                    status="partial",
                )
                final_reels = []
            else:
                _complete_generation(
                    conn,
                    generation_id=generation_id,
                    retrieval_profile="unified",
                    status="failed",
                    error_text="inventory_exhausted",
                )
                final_reels = []
            has_terminal_result = (
                bool(final_reels)
                or has_verified_reservoir
                or provider_cursor_open
                or editorial_reacquisition_available
            )

            usage_records = context.usage()
            usage_payload = generation_usage_payload(
                retry_error=(
                    retrieval_provider_error.as_dict()
                    if retrieval_provider_error is not None
                    else None
                ),
                terminal=True,
            )
            usage_payload = _generation_usage_with_authoritative_release_count(
                usage_payload,
                released_reels=len(final_reels),
            )
            terminal_counters = dict(
                usage_payload.get("counters") or stage_counters
            )
            model_records = [
                row
                for row in usage_records
                if row.get("operation") == "segmentation"
                and bool((row.get("metadata") or {}).get("provider_call"))
            ]
            model_used = str((model_records[-1] if model_records else {}).get("model_used") or "") or None
            quality_degraded = ordering_degraded or any(
                bool(row.get("quality_degraded")) for row in model_records
            )
            terminal_status = (
                "completed"
                if len(final_reels) >= requested_count
                else "partial"
                if has_terminal_result
                else "exhausted"
            )
            def release_generation(release_conn: Any) -> bool:
                _lock_learner_adaptation(
                    release_conn,
                    material_id=material_id,
                    learner_id=learner_id,
                )
                if cancel_if_adaptation_stale(release_conn):
                    return False
                if activate_generation:
                    _activate_generation(
                        release_conn,
                        material_id=material_id,
                        request_key=str(job_row.get("request_key") or ""),
                        generation_id=generation_id,
                        retrieval_profile="unified",
                    )
                append_generation_event(
                    release_conn,
                    job_id=job_id,
                    event_type="final",
                    payload={
                        "reels": final_reels,
                        "generation_id": generation_id if has_terminal_result else None,
                        "authoritative": True,
                        "reconciliation_tail_reel_ids": (
                            reconciliation_tail_reel_ids
                        ),
                    },
                    lease_owner=lease_owner,
                )
                transition_generation_terminal(
                    release_conn,
                    job_id=job_id,
                    status=terminal_status,
                    result_generation_id=(
                        generation_id if has_terminal_result else None
                    ),
                    lease_owner=lease_owner,
                    model_used=model_used,
                    quality_degraded=quality_degraded,
                    usage=usage_payload,
                    error_code=(
                        "inventory_exhausted"
                        if terminal_status == "exhausted"
                        else None
                    ),
                    error_message=(
                        _generation_exhaustion_message(terminal_counters)
                        if terminal_status == "exhausted"
                        else None
                    ),
                    error_detail=(
                        {"counters": terminal_counters}
                        if terminal_status == "exhausted"
                        else None
                    ),
                )
                return True

            released = _run_generation_db_transaction(
                "final_release",
                release_generation,
                retry_should_stop=db_retry_should_stop,
            )
            if not released:
                return
            if recall_preparation is not None:
                logger.info(
                    "recall preparation job_id=%s requested=%d prepared=%d fallback=%d",
                    job_id,
                    recall_preparation["requested"],
                    recall_preparation["prepared"],
                    recall_preparation["fallback"],
                )
    except GenerationCancelledError:
        user_cancelled = False
        with get_conn(transactional=True) as conn:
            user_cancelled = generation_cancellation_requested(conn, job_id)
            if user_cancelled:
                request_generation_cancellation(
                    conn,
                    job_id=job_id,
                    usage=generation_usage_payload(terminal=True),
                )
        if not user_cancelled:
            # Shutdown, deadline, or a lost lease is not a user cancellation.
            # Leave the durable row for expiry/recovery after preserving this
            # attempt's provider exposure.
            checkpoint_yielded_usage()
    except (ClipEngineProviderError, IngestRateLimitedError) as exc:
        provider_exc = (
            ProviderRateLimitError(
                str(exc),
                provider="ingestion",
                operation="platform_rate_limit",
                status_code=429,
                retry_after_sec=exc.retry_after_sec,
                detail=exc.detail,
            )
            if isinstance(exc, IngestRateLimitedError)
            else exc
        )
        logger.warning(
            "generation provider failure job_id=%s error=%s",
            job_id,
            provider_exc.as_dict(),
        )
        provider_usage = generation_usage_payload(
            retry_error=provider_exc.as_dict(),
            terminal=True,
        )
        provider_counters = dict(
            provider_usage.get("counters") or context.counters()
        )
        attempt_count = durable_attempt_count
        if provider_exc.retryable:
            retry_state = _run_generation_db_transaction(
                "provider_failure_requeue",
                lambda retry_conn: requeue_generation_retryable_failure(
                    retry_conn,
                    job_id=job_id,
                    lease_owner=lease_owner,
                    expected_attempt_count=attempt_count,
                    usage=provider_usage,
                    retry_after_sec=provider_exc.retry_after_sec,
                ),
                replay_after_unknown_commit=True,
            )
            if retry_state is not None:
                logger.warning(
                    "requeued retryable generation provider failure "
                    "job_id=%s attempt=%d/%d status=%s",
                    job_id,
                    attempt_count,
                    int(retry_state.get("max_attempts") or 0),
                    str(retry_state.get("status") or ""),
                )
                _wake_generation_worker()
                return
        try:
            _run_generation_db_transaction(
                "provider_failure_terminalization",
                lambda terminal_conn: transition_generation_terminal(
                    terminal_conn,
                    job_id=job_id,
                    status="failed",
                    result_generation_id=generation_id or None,
                    lease_owner=lease_owner,
                    usage=provider_usage,
                    error_code=provider_exc.code,
                    error_message=str(provider_exc),
                    error_detail={
                        **provider_exc.as_dict(),
                        "counters": provider_counters,
                    },
                ),
                retry_should_stop=db_retry_should_stop,
            )
        except JobLeaseLostError:
            logger.info("generation worker lost lease during provider failure job_id=%s", job_id)
            checkpoint_yielded_usage()
            _wake_generation_worker()
    except JobLeaseLostError:
        checkpoint_yielded_usage()
    except Exception as exc:
        logger.exception("generation job failed job_id=%s", job_id)
        failure_usage = generation_usage_payload(terminal=True)
        try:
            _run_generation_db_transaction(
                "failure_terminalization",
                lambda terminal_conn: transition_generation_terminal(
                    terminal_conn,
                    job_id=job_id,
                    status="failed",
                    result_generation_id=generation_id or None,
                    lease_owner=lease_owner,
                    usage=failure_usage,
                    error_code="generation_failed",
                    error_message=str(exc),
                    error_detail={"counters": context.counters()},
                ),
                retry_should_stop=db_retry_should_stop,
            )
        except JobLeaseLostError:
            checkpoint_yielded_usage()
    finally:
        local_stop.set()
        heartbeat_thread.join(timeout=1.0)


def _generation_worker_loop(
    lease_owner: str,
    stop_event: threading.Event,
) -> None:
    worker_id = lease_owner or _generation_worker_id
    while not stop_event.is_set():
        # Clear before the database sweep. A submit committed during the sweep
        # leaves the Event set, so the following wait returns immediately.
        _generation_worker_wake.clear()
        try:
            job_row = _run_generation_db_transaction(
                "job_lease",
                lambda lease_conn: lease_next_job(
                    lease_conn,
                    lease_owner=worker_id,
                    lease_seconds=GENERATION_LEASE_SEC,
                ),
                retry_should_stop=stop_event.is_set,
            )
            if job_row:
                _run_leased_generation_job(job_row, stop_event)
                continue
        except Exception:
            logger.exception("generation worker poll failed")
        if stop_event.is_set():
            break
        wait_seconds = GENERATION_WORKER_POLL_SEC
        try:
            with get_conn() as wait_conn:
                retry_delay = next_queued_retry_delay(wait_conn)
            if retry_delay is not None:
                wait_seconds = min(wait_seconds, max(0.05, retry_delay))
        except Exception:
            logger.exception("generation worker retry-delay lookup failed")
        _generation_worker_wake.wait(wait_seconds)


def _wake_generation_worker() -> None:
    _generation_worker_wake.set()


def _submit_bounded_generation_job(conn, **kwargs: Any) -> tuple[dict[str, Any], bool]:
    request_params = kwargs.get("request_params")
    if not isinstance(request_params, dict):
        raise ValueError("request_params must be a dictionary")
    kwargs["request_params"] = {
        **request_params,
        GENERATION_DURABLE_QUEUE_WAIT_PARAM: True,
    }
    # Admission is still bounded by verified-account, rate, billing, and quota
    # checks. The worker bounds provider concurrency; accepted work stays queued
    # instead of being dropped merely because another topic is still running.
    return submit_generation_job(conn, **kwargs)


def _start_generation_worker() -> None:
    global _generation_worker_id, _generation_worker_ids, _generation_worker_stop
    global _generation_worker_thread, _generation_worker_threads
    with _generation_worker_lock:
        if any(thread.is_alive() for thread in _generation_worker_threads):
            return
        _generation_worker_stop = threading.Event()
        _generation_worker_wake.clear()
        _generation_worker_ids = tuple(
            f"worker-{uuid.uuid4()}" for _index in range(GENERATION_WORKER_COUNT)
        )
        _generation_worker_id = _generation_worker_ids[0]
        _generation_worker_threads = [
            threading.Thread(
                target=_generation_worker_loop,
                args=(worker_id, _generation_worker_stop),
                name=f"reel-generation-worker-{index + 1}",
                daemon=True,
            )
            for index, worker_id in enumerate(_generation_worker_ids)
        ]
        _generation_worker_thread = _generation_worker_threads[0]
        for thread in _generation_worker_threads:
            thread.start()


def _stop_generation_worker() -> None:
    global _generation_worker_thread, _generation_worker_threads
    with _generation_worker_lock:
        threads = list(_generation_worker_threads)
        _generation_worker_stop.set()
        _generation_worker_wake.set()
        join_deadline = time.monotonic() + 5.0
        for thread in threads:
            thread.join(timeout=max(0.0, join_deadline - time.monotonic()))
        _generation_worker_threads = [thread for thread in threads if thread.is_alive()]
        _generation_worker_thread = (
            _generation_worker_threads[0] if _generation_worker_threads else None
        )


def _normalize_datetime_for_api(value: object) -> str | None:
    if isinstance(value, datetime):
        normalized_dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return normalized_dt.astimezone(timezone.utc).isoformat()
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace(" ", "T")
    if re.search(r"[+-]\d{4}$", normalized):
        normalized = f"{normalized[:-5]}{normalized[-5:-2]}:{normalized[-2:]}"
    elif re.search(r"[+-]\d{2}$", normalized):
        normalized = f"{normalized}:00"
    elif re.search(r"\d{2}:\d{2}:\d{2}", normalized) and not re.search(r"(?:Z|[+-]\d{2}:\d{2})$", normalized, re.IGNORECASE):
        # Treat timezone-less DB timestamps as UTC.
        normalized = f"{normalized}Z"
    return normalized


def _compute_updated_label(updated_at_iso: str | None) -> str:
    if not updated_at_iso:
        return "Last Edited: unknown"
    try:
        normalized = updated_at_iso.replace(" ", "T")
        if normalized.endswith(("z", "Z")):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return "Last Edited: unknown"
    delta = datetime.now(timezone.utc) - dt
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "Last Edited: just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"Last Edited: {minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"Last Edited: {hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 7:
        return f"Last Edited: {days} day{'s' if days != 1 else ''} ago"
    if days < 30:
        weeks = days // 7
        return f"Last Edited: {weeks} week{'s' if weeks != 1 else ''} ago"
    if days < 365:
        months = days // 30
        return f"Last Edited: {months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"Last Edited: {years} year{'s' if years != 1 else ''} ago"


def _serialize_community_set(
    row: dict,
    *,
    viewer_vote: str | None = None,
) -> CommunitySetOut:
    """Build the wire-format `CommunitySetOut` for a single row.

    `viewer_vote` is the calling user's vote on this set — one of
    ``"like"`` / ``"dislike"`` / ``None`` — used to fill the
    ``viewer_liked`` / ``viewer_disliked`` response fields. Anonymous
    callers (no session) just pass ``None`` and both flags come back
    False.
    """
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
            try:
                source_url, embed_url = _validate_community_reel_urls(
                    platform=platform,  # type: ignore[arg-type]
                    source_url=source_url,
                    embed_url=embed_url,
                )
            except HTTPException:
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
    dislikes = max(0, _to_int(row.get("dislikes"), 0))
    learners = max(0, _to_int(row.get("learners"), 1))
    created_at = _normalize_datetime_for_api(row.get("created_at"))
    updated_at = _normalize_datetime_for_api(row.get("updated_at")) or created_at
    updated_label = _compute_updated_label(updated_at)
    curator = str(row.get("curator") or "Community member").strip() or "Community member"
    thumbnail_url = str(row.get("thumbnail_url") or "").strip()
    if not thumbnail_url:
        thumbnail_url = "/images/community/ai-systems.svg"

    normalized_vote = str(viewer_vote or "").strip().lower()
    viewer_liked = normalized_vote == "like"
    viewer_disliked = normalized_vote == "dislike"

    return CommunitySetOut(
        id=str(row.get("id") or ""),
        title=str(row.get("title") or "").strip(),
        description=str(row.get("description") or "").strip(),
        tags=tags,
        reels=reels,
        reel_count=reel_count,
        curator=curator,
        likes=likes,
        dislikes=dislikes,
        viewer_liked=viewer_liked,
        viewer_disliked=viewer_disliked,
        learners=learners,
        updated_label=updated_label,
        updated_at=updated_at,
        created_at=created_at,
        thumbnail_url=thumbnail_url,
        featured=bool(_to_int(row.get("featured"), 0)),
    )


def _fetch_viewer_votes_by_set_id(
    conn: Any,
    *,
    account_id: str | None,
    set_ids: Iterable[str],
) -> dict[str, str]:
    """Return ``{set_id: vote}`` for the given account across ``set_ids``.

    Returns an empty dict if ``account_id`` is falsy or no rows match.
    Used by the list endpoints so each `CommunitySetOut` knows whether
    the calling viewer has already voted on it.
    """
    if not account_id:
        return {}
    unique_ids = [sid for sid in {str(sid) for sid in set_ids if sid}]
    if not unique_ids:
        return {}
    placeholders = ", ".join(["?"] * len(unique_ids))
    rows = fetch_all(
        conn,
        f"""
        SELECT set_id, vote
        FROM community_set_votes
        WHERE account_id = ? AND set_id IN ({placeholders})
        """,
        (str(account_id), *unique_ids),
    )
    return {
        str(row.get("set_id") or ""): str(row.get("vote") or "").strip().lower()
        for row in rows
        if row.get("set_id")
    }


def _try_get_community_account(conn: Any, request: Request) -> dict[str, object] | None:
    """Optional variant of `_require_authenticated_community_account`.

    Returns ``None`` instead of raising when the session header is
    missing, malformed, or expired. Used by read endpoints that want to
    personalize their response (e.g. filling `viewer_liked`) without
    forcing the caller to be signed in.
    """
    try:
        return _require_authenticated_community_account(conn, request)
    except HTTPException:
        return None


def _resolve_learner_identity(
    conn: Any, request: Request, *, required: bool = True,
) -> str | None:
    """Prefer an authenticated account; otherwise use the hashed owner key."""
    account = _try_get_community_account(conn, request)
    if account and account.get("id"):
        return f"account:{account['id']}"
    owner_hash = _community_owner_hash_from_request_optional(request)
    if owner_hash:
        return f"owner:{owner_hash}"
    if required:
        raise HTTPException(status_code=401, detail="Missing client identity header.")
    return None


def _serialize_community_history_item(
    row: dict, *, assessment_stats: dict[str, Any] | None = None
) -> CommunityHistoryItemOut:
    generation_mode = str(row.get("generation_mode") or "").strip().lower()
    if generation_mode not in {"slow", "fast"}:
        generation_mode = "slow"
    source = str(row.get("source") or "").strip().lower()
    if source not in {"search", "community"}:
        source = "search"
    feed_query_raw = str(row.get("feed_query") or "").strip()
    active_index = _to_int(row.get("active_index"), -1)
    active_reel_id_raw = str(row.get("active_reel_id") or "").strip()
    recall: dict[str, Any] | None = None
    raw_recall = row.get("recall_json")
    if isinstance(raw_recall, dict):
        recall = raw_recall
    elif raw_recall:
        try:
            parsed_recall = json.loads(str(raw_recall))
            if isinstance(parsed_recall, dict) and parsed_recall:
                recall = parsed_recall
        except (TypeError, json.JSONDecodeError):
            recall = None
    recent_recall_accuracy = row.get("recent_recall_accuracy")
    rolling_recall_accuracy = row.get("rolling_recall_accuracy")
    if recall:
        if recent_recall_accuracy is None:
            recent_recall_accuracy = recall.get("recent_accuracy")
        if rolling_recall_accuracy is None:
            rolling_recall_accuracy = recall.get("rolling_accuracy")
    if assessment_stats and int(assessment_stats.get("completed_checks") or 0) > 0:
        recall = dict(recall or {})
        authoritative_fields = {
            "recent_score": assessment_stats.get("recent_correct"),
            "recent_question_count": assessment_stats.get("recent_total"),
            "recent_accuracy": assessment_stats.get("recent_accuracy"),
            "rolling_accuracy": assessment_stats.get("rolling_accuracy"),
            "completed_at": assessment_stats.get("last_completed_at"),
        }
        recall.update(
            {key: value for key, value in authoritative_fields.items() if value is not None}
        )
        recent_recall_accuracy = assessment_stats.get("recent_accuracy")
        rolling_recall_accuracy = assessment_stats.get("rolling_accuracy")
    return CommunityHistoryItemOut(
        material_id=str(row.get("material_id") or "").strip(),
        title=str(row.get("title") or "").strip() or "New Study Session",
        updated_at=max(0, _to_int(row.get("updated_at"), 0)),
        starred=bool(_to_int(row.get("starred"), 0)),
        generation_mode=generation_mode,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        feed_query=feed_query_raw or None,
        active_index=active_index if active_index >= 0 else None,
        active_reel_id=active_reel_id_raw or None,
        recall=recall,
        recent_recall_accuracy=recent_recall_accuracy,
        rolling_recall_accuracy=rolling_recall_accuracy,
    )


def _normalize_community_history_items(payload_items) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen_material_ids: set[str] = set()
    ordered_items = sorted(payload_items, key=lambda item: max(0, int(item.updated_at)), reverse=True)
    for item in ordered_items:
        material_id = str(item.material_id or "").strip()
        if not material_id or material_id in seen_material_ids:
            continue
        seen_material_ids.add(material_id)
        title = str(item.title or "").strip() or "New Study Session"
        generation_mode = str(item.generation_mode or "").strip().lower()
        if generation_mode not in {"slow", "fast"}:
            generation_mode = "slow"
        source = str(item.source or "").strip().lower()
        if source not in {"search", "community"}:
            source = "search"
        feed_query_raw = str(item.feed_query or "").strip()
        active_index = max(0, int(item.active_index)) if item.active_index is not None else None
        active_reel_id_raw = str(item.active_reel_id or "").strip()
        recall_payload = item.recall.model_dump(exclude_none=True) if item.recall else {}
        if item.recent_recall_accuracy is not None and "recent_accuracy" not in recall_payload:
            recall_payload["recent_accuracy"] = item.recent_recall_accuracy
        if item.rolling_recall_accuracy is not None and "rolling_accuracy" not in recall_payload:
            recall_payload["rolling_accuracy"] = item.rolling_recall_accuracy
        normalized.append(
            {
                "material_id": material_id,
                "title": title,
                "updated_at": max(0, int(item.updated_at)),
                "starred": 1 if item.starred else 0,
                "generation_mode": generation_mode,
                "source": source,
                "feed_query": feed_query_raw or None,
                "active_index": active_index,
                "active_reel_id": active_reel_id_raw or None,
                "recall_json": dumps_json(recall_payload),
            }
        )
        if len(normalized) >= MAX_COMMUNITY_HISTORY_ITEMS:
            break
    return normalized


@app.get("/")
def root() -> dict:
    return {"ok": True, "service": "StudyReels API", "health": "/api/health"}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/admin/health")
def admin_health() -> dict:
    generation_workers_alive = sum(
        thread.is_alive() for thread in _generation_worker_threads
    )
    generation_pool_ready = (
        len(_generation_worker_threads) == GENERATION_WORKER_COUNT
        and generation_workers_alive == GENERATION_WORKER_COUNT
        and not _generation_worker_stop.is_set()
    )
    text_llm = llm_router.text_llm_status(
        gemini_model=material_intelligence_service.model,
    )
    embedding_backend = embedding_service.backend_name
    return {
        "ok": generation_pool_ready,
        "generation_worker_alive": generation_pool_ready,
        "generation_worker_count": GENERATION_WORKER_COUNT,
        "generation_workers_alive": generation_workers_alive,
        "generation_worker_recovery_sec": GENERATION_WORKER_POLL_SEC,
        "supadata_configured": bool(os.getenv("SUPADATA_API_KEY", "").strip()),
        "gemini_primary_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "gemini_chat_configured": bool(os.getenv("GEMINI_API_KEY_2", "").strip()),
        "gemini_clip_selector_model": pipeline_config.SEGMENT_PRO_MODEL,
        # Preserve the health field while reporting only the selector's actual
        # optional failover. Hosted production leaves it disabled.
        "gemini_fallback_model": (
            pipeline_config.SEGMENT_FLASH_FALLBACK_MODEL or None
        ),
        "text_llm_available": bool(text_llm["available"]),
        "text_llm_provider": text_llm["provider"],
        "text_llm_providers": text_llm["providers"],
        "chat_model": text_llm["model"] or material_intelligence_service.model,
        "embedding_backend": embedding_backend,
    }


@app.post("/api/community/auth/send-signup-verification", response_model=CommunitySendSignupVerificationResponse)
def send_signup_verification_email(request: Request, payload: CommunitySendSignupVerificationRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    email, email_normalized = _normalize_community_email(payload.email)
    username = str(payload.username or "").strip() or None
    verification_required = _community_email_verification_required()
    if verification_required:
        _require_hosted_verification_delivery_available()
    with get_conn(transactional=True) as conn:
        existing_email = fetch_one(
            conn,
            "SELECT id FROM community_accounts WHERE email_normalized = ? LIMIT 1",
            (email_normalized,),
        )
        if existing_email:
            raise HTTPException(status_code=409, detail=COMMUNITY_REGISTER_CONFLICT_DETAIL)
        if not verification_required:
            return CommunitySendSignupVerificationResponse(
                email=email,
                verification_required=False,
                verified=True,
                verification_code_debug=None,
            )
        owner_key_hash = _community_owner_hash_from_request(request)
        verification_code_debug = _issue_community_signup_verification_code(
            conn,
            owner_key_hash=owner_key_hash,
            email=email,
            email_normalized=email_normalized,
            username=username,
        )
    return CommunitySendSignupVerificationResponse(
        email=email,
        verification_required=True,
        verified=False,
        verification_code_debug=verification_code_debug,
    )


@app.post("/api/community/auth/verify-signup-email", response_model=CommunityVerifySignupEmailResponse)
def verify_signup_email(request: Request, payload: CommunityVerifySignupEmailRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    email, email_normalized = _normalize_community_email(payload.email)
    verification_code = str(payload.code or "").strip()
    if not _community_email_verification_required():
        return CommunityVerifySignupEmailResponse(email=email, verified=True)
    with get_conn(transactional=True) as conn:
        owner_key_hash = _community_owner_hash_from_request(request)
        verification = _load_community_signup_verification(
            conn,
            owner_key_hash=owner_key_hash,
            email_normalized=email_normalized,
        )
        if not verification:
            raise HTTPException(status_code=400, detail="Send a verification code before trying to verify this email.")
        if _community_signup_email_is_verified(verification):
            return CommunityVerifySignupEmailResponse(
                email=str(verification.get("email") or email).strip() or email,
                verified=True,
            )
        _check_rate_limit_key(
            f"community-verify-signup:{owner_key_hash}:{email_normalized}",
            limit=COMMUNITY_VERIFY_PER_ACCOUNT_RATE_LIMIT,
            window_sec=RATE_LIMIT_WINDOW_SEC,
        )
        expected_hash = str(verification.get("verification_code_hash") or "").strip()
        expires_at = _parse_optional_iso_datetime(verification.get("verification_expires_at"))
        if not expected_hash or expires_at is None or expires_at <= datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Verification code expired. Request a new code.")
        if not secrets.compare_digest(expected_hash, _hash_community_verification_code(verification_code)):
            raise HTTPException(status_code=400, detail="Verification code is incorrect.")
        verified_at = now_iso()
        execute_modify(
            conn,
            """
            UPDATE community_signup_email_verifications
            SET verified_at = ?, verification_code_hash = NULL, verification_expires_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (verified_at, verified_at, str(verification.get("id") or "")),
        )
    return CommunityVerifySignupEmailResponse(email=email, verified=True)


@app.post("/api/community/auth/register", response_model=CommunityAuthSessionResponse, status_code=201)
def register_community_account(request: Request, payload: CommunityAuthRegisterRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    username, username_normalized = _normalize_community_username(payload.username)
    email, email_normalized = _normalize_community_email(payload.email)
    password = _normalize_community_password(payload.password)
    legacy_claim_owner_key_hash = _community_owner_hash_from_request_optional(request)
    verification_required = _community_email_verification_required()
    if verification_required:
        _require_hosted_verification_delivery_available()
    with get_conn(transactional=True) as conn:
        existing_identity = fetch_one(
            conn,
            """
            SELECT id
            FROM community_accounts
            WHERE username_normalized = ?
               OR email_normalized = ?
            LIMIT 1
            """,
            (username_normalized, email_normalized),
        )
        if existing_identity:
            raise HTTPException(status_code=409, detail=COMMUNITY_REGISTER_CONFLICT_DETAIL)
        verified_at = now_iso()
        if verification_required:
            owner_key_hash = legacy_claim_owner_key_hash or _community_owner_hash_from_request(request)
            legacy_claim_owner_key_hash = owner_key_hash
            pending_signup_verification = _load_community_signup_verification(
                conn,
                owner_key_hash=owner_key_hash,
                email_normalized=email_normalized,
            )
            if not _community_signup_email_is_verified(pending_signup_verification):
                raise HTTPException(status_code=403, detail="Verify your email before creating the account.")
            verified_at = str(pending_signup_verification.get("verified_at") or "").strip() or verified_at
        account_id = str(uuid.uuid4())
        salt_hex = secrets.token_hex(16)
        timestamp = now_iso()
        try:
            insert(
                conn,
                "community_accounts",
                {
                    "id": account_id,
                    "username": username,
                    "username_normalized": username_normalized,
                    "email": email,
                    "email_normalized": email_normalized,
                    "password_hash": _hash_community_password(password, salt_hex),
                    "password_salt": salt_hex,
                    "verified_at": verified_at,
                    "verification_code_hash": None,
                    "verification_expires_at": None,
                    "legacy_claim_owner_key_hash": legacy_claim_owner_key_hash,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )
        except DatabaseIntegrityError as exc:
            raise HTTPException(status_code=409, detail=COMMUNITY_REGISTER_CONFLICT_DETAIL) from exc
        session_token = _create_community_session(conn, account_id)
        verification_code_debug = None
        claimed_legacy_sets = _claim_legacy_community_sets_for_account(
            conn,
            request,
            account_id=account_id,
            legacy_claim_owner_key_hash=legacy_claim_owner_key_hash,
        )
        if verification_required:
            execute_modify(
                conn,
                "DELETE FROM community_signup_email_verifications WHERE id = ?",
                (_community_signup_verification_id(owner_key_hash=legacy_claim_owner_key_hash or _community_owner_hash_from_request(request), email_normalized=email_normalized),),
            )
        account = {
            "id": account_id,
            "username": username,
            "email": email,
            "verified_at": verified_at,
        }
    send_welcome_email(email=email, username=username)
    return CommunityAuthSessionResponse(
        account=_community_account_out(account),
        session_token=session_token,
        claimed_legacy_sets=max(0, claimed_legacy_sets),
        verification_required=False,
        verification_code_debug=verification_code_debug,
    )


@app.post("/api/community/auth/login", response_model=CommunityAuthSessionResponse)
def login_community_account(request: Request, payload: CommunityAuthLoginRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    _, username_normalized = _normalize_community_username(payload.username)
    password = _normalize_community_password(payload.password)
    _check_rate_limit_key(
        f"community-login-user:{username_normalized}:{_client_ip(request)}",
        limit=COMMUNITY_LOGIN_PER_USERNAME_RATE_LIMIT,
        window_sec=RATE_LIMIT_WINDOW_SEC,
    )
    claimed_legacy_sets = 0
    verification_code_debug = None
    with get_conn(transactional=True) as conn:
        account = fetch_one(
            conn,
            """
            SELECT
                id,
                username,
                email,
                username_normalized,
                password_hash,
                password_salt,
                verified_at,
                verification_code_hash,
                verification_expires_at,
                legacy_claim_owner_key_hash
            FROM community_accounts
            WHERE username_normalized = ?
            LIMIT 1
            """,
            (username_normalized,),
        )
        if not account or not _verify_community_password(password, str(account.get("password_salt") or ""), str(account.get("password_hash") or "")):
            raise HTTPException(status_code=401, detail="Incorrect username or password.")
        account_id = str(account["id"])
        verification_required = _community_email_verification_required()
        if not _community_account_is_verified(account):
            email = str(account.get("email") or "").strip()
            if not email:
                raise HTTPException(status_code=409, detail="This account is missing a verification email.")
            if verification_required:
                _require_hosted_verification_delivery_available()
        session_token = _create_community_session(conn, account_id)
        if not _community_account_is_verified(account):
            if verification_required:
                verification_code_debug = _issue_community_verification_code(
                    conn,
                    account_id=account_id,
                    email=str(account.get("email") or "").strip(),
                    username=str(account.get("username") or "").strip(),
                )
            else:
                account, claimed_legacy_sets = _auto_verify_community_account_if_allowed(
                    conn,
                    request,
                    account,
                    claim_legacy_sets=True,
                )
        else:
            claimed_legacy_sets = _claim_legacy_community_sets_for_account(
                conn,
                request,
                account_id=account_id,
                legacy_claim_owner_key_hash=account.get("legacy_claim_owner_key_hash"),
            )
    if not _community_account_is_verified(account):
        return CommunityAuthSessionResponse(
            account=_community_account_out(account),
            session_token=session_token,
            claimed_legacy_sets=0,
            verification_required=verification_required,
            verification_code_debug=verification_code_debug,
        )
    return CommunityAuthSessionResponse(
        account=_community_account_out(account),
        session_token=session_token,
        claimed_legacy_sets=max(0, claimed_legacy_sets),
        verification_required=False,
        verification_code_debug=None,
    )


@app.get("/api/community/auth/me", response_model=CommunityAuthMeResponse)
def get_community_account_me(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        account, _ = _auto_verify_community_account_if_allowed(conn, request, account, claim_legacy_sets=True)
    return CommunityAuthMeResponse(account=_community_account_out(account))


@app.post("/api/community/auth/verify", response_model=CommunityVerifyAccountResponse)
def verify_community_account(request: Request, payload: CommunityVerifyAccountRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    verification_code = str(payload.code or "").strip()
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        account_id = str(account["id"])
        full_account = fetch_one(
            conn,
            """
            SELECT
                id,
                username,
                email,
                verified_at,
                verification_code_hash,
                verification_expires_at,
                legacy_claim_owner_key_hash
            FROM community_accounts
            WHERE id = ?
            LIMIT 1
            """,
            (account_id,),
        )
        if not full_account:
            raise HTTPException(status_code=404, detail="Community account not found.")
        full_account, auto_claimed_legacy_sets = _auto_verify_community_account_if_allowed(
            conn,
            request,
            full_account,
            claim_legacy_sets=True,
        )
        if auto_claimed_legacy_sets > 0:
            return CommunityVerifyAccountResponse(
                account=_community_account_out(full_account),
                claimed_legacy_sets=max(0, auto_claimed_legacy_sets),
            )
        if _community_account_is_verified(full_account):
            claimed_legacy_sets = _claim_legacy_community_sets_for_account(
                conn,
                request,
                account_id=account_id,
                legacy_claim_owner_key_hash=full_account.get("legacy_claim_owner_key_hash"),
            )
            return CommunityVerifyAccountResponse(
                account=_community_account_out(full_account),
                claimed_legacy_sets=max(0, claimed_legacy_sets),
            )
        # Per-account brute-force protection: 6-digit codes have only 1M keyspace,
        # so limit verification attempts to 5 per window per account (on top of IP limit).
        _check_rate_limit_key(
            f"community-verify-account:{account_id}",
            limit=COMMUNITY_VERIFY_PER_ACCOUNT_RATE_LIMIT,
            window_sec=RATE_LIMIT_WINDOW_SEC,
        )
        expected_hash = str(full_account.get("verification_code_hash") or "").strip()
        expires_at = _parse_optional_iso_datetime(full_account.get("verification_expires_at"))
        if not expected_hash or expires_at is None or expires_at <= datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="Verification code expired. Request a new code.")
        if not secrets.compare_digest(expected_hash, _hash_community_verification_code(verification_code)):
            raise HTTPException(status_code=400, detail="Verification code is incorrect.")
        verified_at = now_iso()
        execute_modify(
            conn,
            """
            UPDATE community_accounts
            SET verified_at = ?, verification_code_hash = NULL, verification_expires_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (verified_at, verified_at, account_id),
        )
        full_account["verified_at"] = verified_at
        full_account["verification_code_hash"] = None
        full_account["verification_expires_at"] = None
        claimed_legacy_sets = _claim_legacy_community_sets_for_account(
            conn,
            request,
            account_id=account_id,
            legacy_claim_owner_key_hash=full_account.get("legacy_claim_owner_key_hash"),
        )
        return CommunityVerifyAccountResponse(
            account=_community_account_out(full_account),
            claimed_legacy_sets=max(0, claimed_legacy_sets),
        )


@app.post("/api/community/auth/resend-verification", response_model=CommunityResendVerificationResponse)
def resend_community_verification(request: Request):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        account_id = str(account["id"])
        full_account = fetch_one(
            conn,
            """
            SELECT
                id,
                username,
                email,
                verified_at
            FROM community_accounts
            WHERE id = ?
            LIMIT 1
            """,
            (account_id,),
        )
        if not full_account:
            raise HTTPException(status_code=404, detail="Community account not found.")
        full_account, _ = _auto_verify_community_account_if_allowed(conn, request, full_account, claim_legacy_sets=True)
        if not _community_email_verification_required():
            return CommunityResendVerificationResponse(account=_community_account_out(full_account), verification_code_debug=None)
        if _community_account_is_verified(full_account):
            return CommunityResendVerificationResponse(account=_community_account_out(full_account), verification_code_debug=None)
        _require_hosted_verification_delivery_available()
        email = str(full_account.get("email") or "").strip()
        if not email:
            raise HTTPException(status_code=409, detail="This account is missing a verification email.")
        verification_code_debug = _issue_community_verification_code(
            conn,
            account_id=account_id,
            email=email,
            username=str(full_account.get("username") or "").strip(),
        )
        return CommunityResendVerificationResponse(
            account=_community_account_out(full_account),
            verification_code_debug=verification_code_debug,
        )


@app.post("/api/community/auth/change-email", response_model=CommunityChangeEmailResponse)
def change_community_verification_email(request: Request, payload: CommunityChangeEmailRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    email, email_normalized = _normalize_community_email(payload.email)
    current_password = _normalize_community_password(payload.current_password)
    verification_required = _community_email_verification_required()
    if verification_required:
        _require_hosted_verification_delivery_available()
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        account_id = str(account["id"])
        full_account = fetch_one(
            conn,
            """
            SELECT
                id,
                username,
                email,
                email_normalized,
                password_hash,
                password_salt,
                verified_at
            FROM community_accounts
            WHERE id = ?
            LIMIT 1
            """,
            (account_id,),
        )
        if not full_account:
            raise HTTPException(status_code=404, detail="Community account not found.")
        if verification_required and _community_account_is_verified(full_account):
            raise HTTPException(status_code=403, detail="Verified accounts cannot change email from this screen.")
        if not _verify_community_password(
            current_password,
            str(full_account.get("password_salt") or ""),
            str(full_account.get("password_hash") or ""),
        ):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")

        previous_email_normalized = str(full_account.get("email_normalized") or "").strip().lower()
        if email_normalized != previous_email_normalized:
            existing_email = fetch_one(
                conn,
                "SELECT id FROM community_accounts WHERE email_normalized = ? AND id <> ? LIMIT 1",
                (email_normalized, account_id),
            )
            if existing_email:
                raise HTTPException(status_code=409, detail=COMMUNITY_CHANGE_EMAIL_CONFLICT_DETAIL)

        try:
            execute_modify(
                conn,
                """
                UPDATE community_accounts
                SET email = ?, email_normalized = ?, verified_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (email, email_normalized, now_iso(), account_id),
            )
        except Exception as exc:
            lower_error = str(exc).lower()
            if "email_normalized" in lower_error or "unique" in lower_error:
                raise HTTPException(status_code=409, detail=COMMUNITY_CHANGE_EMAIL_CONFLICT_DETAIL) from exc
            raise
        verification_code_debug = None
        if verification_required:
            verification_code_debug = _issue_community_verification_code(
                conn,
                account_id=account_id,
                email=email,
                username=str(full_account.get("username") or "").strip(),
            )
        else:
            verified_at = now_iso()
            execute_modify(
                conn,
                """
                UPDATE community_accounts
                SET verified_at = ?, verification_code_hash = NULL, verification_expires_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (verified_at, verified_at, account_id),
            )
            full_account["verified_at"] = verified_at

        full_account["email"] = email
        full_account["email_normalized"] = email_normalized
        if verification_required:
            full_account["verified_at"] = None
        return CommunityChangeEmailResponse(
            account=_community_account_out(full_account),
            verification_code_debug=verification_code_debug,
        )


@app.post("/api/community/auth/logout", status_code=204)
def logout_community_account(request: Request):
    raw_token = str(request.headers.get(COMMUNITY_SESSION_HEADER) or "").strip()
    if not raw_token:
        return Response(status_code=204)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    with get_conn(transactional=True) as conn:
        execute_modify(conn, "DELETE FROM community_sessions WHERE token_hash = ?", (token_hash,))
    return Response(status_code=204)


@app.post("/api/community/auth/change-password", status_code=200)
def change_community_password(request: Request, payload: CommunityChangePasswordRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    current_password = _normalize_community_password(payload.current_password)
    new_password = _normalize_community_password(payload.new_password)
    with get_conn(transactional=True) as conn:
        account_row = _require_authenticated_community_account(conn, request)
        account_id = str(account_row["id"])
        full_account = fetch_one(
            conn,
            "SELECT password_hash, password_salt FROM community_accounts WHERE id = ? LIMIT 1",
            (account_id,),
        )
        if not full_account or not _verify_community_password(
            current_password,
            str(full_account.get("password_salt") or ""),
            str(full_account.get("password_hash") or ""),
        ):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")
        new_salt_hex = secrets.token_hex(16)
        execute_modify(
            conn,
            "UPDATE community_accounts SET password_hash = ?, password_salt = ?, updated_at = ? WHERE id = ?",
            (_hash_community_password(new_password, new_salt_hex), new_salt_hex, now_iso(), account_id),
        )
        # Invalidate all other sessions so other devices must re-authenticate.
        session_token = _community_session_token_from_request(request)
        current_token_hash = hashlib.sha256(session_token.encode("utf-8")).hexdigest()
        execute_modify(
            conn,
            "DELETE FROM community_sessions WHERE account_id = ? AND token_hash <> ?",
            (account_id, current_token_hash),
        )
    return {"status": "ok"}


@app.get("/api/billing/plans", response_model=BillingPlansResponse)
def get_billing_plans():
    return plans_payload()


@app.get("/api/billing/status", response_model=BillingStatusResponse)
def get_billing_status(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_verified_provider_account(conn, request)
        return billing_status(conn, str(account["id"]))


@app.post(
    "/api/billing/stripe/checkout",
    response_model=BillingRedirectResponse,
)
def start_stripe_checkout(request: Request, payload: BillingCheckoutRequest):
    try:
        with get_conn(transactional=True) as conn:
            account = _require_verified_provider_account(conn, request)
            return {
                "url": create_stripe_checkout(
                    conn,
                    account=dict(account),
                    plan_code=payload.plan,
                )
            }
    except DuplicateSubscriptionError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "subscription_already_active",
                "provider": "stripe",
                "message": str(exc),
            },
        ) from exc
    except BillingAccountNotFoundError as exc:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "verified_account_required",
                "message": "Sign in to a verified ReelAI account to start a new search.",
            },
        ) from exc
    except BillingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/api/billing/stripe/portal",
    response_model=BillingRedirectResponse,
)
def start_stripe_portal(request: Request):
    try:
        with get_conn(transactional=True) as conn:
            account = _require_verified_provider_account(conn, request)
            return {"url": create_stripe_portal(conn, account_id=str(account["id"]))}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BillingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/billing/stripe/webhook")
async def stripe_billing_webhook(request: Request):
    signature = str(request.headers.get("Stripe-Signature") or "").strip()
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header.")
    try:
        event = construct_stripe_event(await request.body(), signature)
        with get_conn(transactional=True) as conn:
            processed = process_stripe_event(conn, event)
    except BillingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (BillingVerificationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook.") from exc
    return {"received": True, "processed": processed}


@app.post("/api/community/auth/delete-account", status_code=204)
def delete_community_account(request: Request, payload: CommunityDeleteAccountRequest):
    _enforce_rate_limit(request, "community-auth", limit=COMMUNITY_AUTH_RATE_LIMIT_PER_WINDOW)
    current_password = _normalize_community_password(payload.current_password)
    with get_conn(transactional=True) as conn:
        account_row = _require_authenticated_community_account(conn, request)
        account_id = str(account_row["id"])
        full_account = fetch_one(
            conn,
            "SELECT password_hash, password_salt FROM community_accounts WHERE id = ? LIMIT 1",
            (account_id,),
        )
        if not full_account or not _verify_community_password(
            current_password,
            str(full_account.get("password_salt") or ""),
            str(full_account.get("password_hash") or ""),
        ):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")
        try:
            lock_billing_account(conn, account_id)
            cancel_stripe_for_account(conn, account_id)
        except BillingAccountNotFoundError as exc:
            raise HTTPException(status_code=401, detail="Session expired. Sign in again.") from exc
        except BillingConfigurationError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "stripe_cancellation_unavailable",
                    "message": "Stripe billing could not be cancelled. Try account deletion again.",
                },
            ) from exc
        execute_modify(conn, "DELETE FROM community_sets WHERE owner_account_id = ?", (account_id,))
        execute_modify(conn, "DELETE FROM community_material_history WHERE account_id = ?", (account_id,))
        execute_modify(conn, "DELETE FROM community_account_settings WHERE account_id = ?", (account_id,))
        execute_modify(
            conn,
            "DELETE FROM api_idempotency_records WHERE learner_id = ? OR learner_id = ?",
            (account_id, f"account:{account_id}"),
        )
        execute_modify(conn, "DELETE FROM community_sessions WHERE account_id = ?", (account_id,))
        execute_modify(conn, "DELETE FROM community_accounts WHERE id = ?", (account_id,))
    return Response(status_code=204)


@app.post("/api/material", response_model=MaterialResponse)
async def create_material(
    request: Request,
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    subject_tag: str | None = Form(default=None),
    knowledge_level: str | None = Form(default=None),
):
    _enforce_rate_limit(request, "material", limit=MATERIAL_RATE_LIMIT_PER_WINDOW)
    if subject_tag is not None and not subject_tag.strip() and not file and not (text or "").strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "blank_retrieval_topic", "message": "Topic must contain non-whitespace text."},
        )
    if not file and not text and not subject_tag:
        raise HTTPException(status_code=400, detail="Provide at least one of: topic, text, or file")
    from .services.knowledge_level import normalize_knowledge_level
    try:
        normalized_level = normalize_knowledge_level(knowledge_level)
    except ValueError:
        raise HTTPException(status_code=422, detail="knowledge_level must be beginner, intermediate, or advanced")
    def resolve_material_identity(identity_conn: Any) -> tuple[str, str]:
        provider_account = _require_verified_provider_account(identity_conn, request)
        account_id = str(provider_account["id"])
        resolved_learner_id = _resolve_learner_identity(
            identity_conn,
            request,
            required=False,
        ) or f"account:{account_id}"
        return account_id, resolved_learner_id

    quota_account_id, learner_id = _run_generation_db_transaction(
        "material_identity",
        resolve_material_identity,
        replay_after_unknown_commit=True,
    )
    try:
        idempotency_key = normalize_idempotency_key(
            request.headers.get("Idempotency-Key")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if billing_enforcement_enabled() and not idempotency_key:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "idempotency_key_required",
                "message": "Idempotency-Key is required for a new material search.",
            },
        )

    source_parts: list[str] = []
    analysis_source_parts: list[str] = []
    source_type = "mixed"
    source_path = None
    file_content: bytes | None = None
    file_name = ""

    if file:
        file_content = await file.read()
        file_name = file.filename or "material"
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            file_text = extract_text_from_file(file.filename or "upload.txt", file_content)
        except ParseError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if file_text.strip():
            source_parts.append(file_text)
            analysis_source_parts.append(file_text)
        source_type = "file"

    if text and text.strip():
        source_parts.append(text)
        analysis_source_parts.append(text)
        if source_type != "file":
            source_type = "text"

    if subject_tag and subject_tag.strip():
        source_parts.append(f"Topic: {subject_tag.strip()}")
        if source_type not in {"file", "text"}:
            source_type = "topic"

    structured_source_text = "\n\n".join(source_parts)
    structured_analysis_text = (
        "\n\n".join(analysis_source_parts) or structured_source_text
    )
    raw_text = normalize_whitespace(structured_source_text)
    if len(raw_text) < 3:
        raise HTTPException(status_code=400, detail="Input is too short")

    chunks = chunk_text(raw_text)
    if SERVERLESS_MODE and len(chunks) > 64:
        chunks = chunks[:64]

    material_id = str(uuid.uuid4())
    created_at = now_iso()

    idempotency_learner = str(learner_id or "anonymous")
    reservation_owned = False
    reservation_attempt_token = ""
    reservation_reclaimed = False
    fingerprint = ""
    if idempotency_key:
        fingerprint = build_idempotency_fingerprint(
            {
                "version": 1,
                "subject_tag": str(subject_tag or ""),
                "text": str(text or ""),
                "knowledge_level": normalized_level,
                "source_type": source_type,
                "file_name": file_name,
                "file_sha256": (
                    hashlib.sha256(file_content).hexdigest()
                    if file_content is not None
                    else ""
                ),
            }
        )
        owned_reservation: Any | None = None

        def reserve_material_idempotency(idempotency_conn: Any) -> Any:
            nonlocal owned_reservation
            reservation = reserve_idempotency_key(
                    idempotency_conn,
                    scope="material",
                    learner_id=idempotency_learner,
                    raw_key=idempotency_key,
                    fingerprint=fingerprint,
                    resource_id=material_id,
                    stale_after_seconds=30 * 60,
                )
            if reservation.owner:
                owned_reservation = reservation
            elif (
                owned_reservation is not None
                and reservation.status == "in_progress"
                and reservation.resource_id == owned_reservation.resource_id
            ):
                # The first transaction committed but its acknowledgement was
                # lost.  Continue with the attempt token created by that exact
                # request instead of treating our own replay as a competitor.
                return owned_reservation
            return reservation

        try:
            reservation = _run_generation_db_transaction(
                "material_idempotency_reserve",
                reserve_material_idempotency,
                replay_after_unknown_commit=True,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "idempotency_key_reused",
                    "message": str(exc),
                },
            ) from exc
        material_id = reservation.resource_id
        reservation_owned = reservation.owner
        reservation_attempt_token = str(reservation.attempt_token or "")
        reservation_reclaimed = reservation.reclaimed
        if not reservation.owner:
            if reservation.status == "completed" and reservation.response:
                return reservation.response
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "idempotency_in_progress",
                    "message": "This material submission is already being processed.",
                },
                headers={"Retry-After": "1"},
            )

    quota_operation_key = f"material:{material_id}"
    if quota_account_id:
        try:
            _run_generation_db_transaction(
                "material_quota_reserve",
                lambda quota_conn: _reserve_search_or_http(
                    quota_conn,
                    account_id=quota_account_id,
                    operation_key=quota_operation_key,
                    surface="material",
                    material_id=material_id,
                ),
                replay_after_unknown_commit=True,
            )
        except Exception:
            if idempotency_key and reservation_owned:
                _run_generation_db_transaction(
                    "material_idempotency_release_after_quota_failure",
                    lambda idempotency_conn: release_idempotency_key(
                        idempotency_conn,
                        scope="material",
                        learner_id=idempotency_learner,
                        raw_key=idempotency_key,
                        resource_id=material_id,
                        attempt_token=reservation_attempt_token,
                    ),
                    replay_after_unknown_commit=True,
                )
            raise

    try:
        with get_conn() as provider_conn:
            concept_limit = 6 if SERVERLESS_MODE else 12
            concepts, _objectives = material_intelligence_service.extract_concepts_and_objectives(
                provider_conn,
                structured_analysis_text,
                subject_tag=subject_tag,
                max_concepts=concept_limit,
            )

            concept_embeddings = []
            for concept in concepts:
                concept_text = (
                    f"{concept['title']}. Keywords: {' '.join(concept['keywords'])}. Summary: {concept['summary']}"
                )
                concept_embeddings.append(
                    embedding_service.embed_texts(provider_conn, [concept_text])[0]
                )
            chunk_embeddings = (
                embedding_service.embed_texts(provider_conn, chunks) if chunks else []
            )

        chunk_records = [
            {
                "id": str(uuid.uuid4()),
                "chunk_index": index,
                "text": chunk,
                "embedding_json": dumps_json(embedding.tolist()),
            }
            for index, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings))
        ]
        if file_content is not None and not reservation_owned:
            source_path = storage.save_bytes(file_content, file_name)
        response_payload = {
            "material_id": material_id,
            "extracted_concepts": concepts,
        }
        publication_body_completed = False

        def persist_material(persist_conn: Any) -> dict[str, Any]:
            nonlocal publication_body_completed, source_path
            if publication_body_completed:
                existing_material = fetch_one(
                    persist_conn,
                    "SELECT id FROM materials WHERE id = ?",
                    (material_id,),
                )
                if existing_material and idempotency_key:
                    replay = reserve_idempotency_key(
                        persist_conn,
                        scope="material",
                        learner_id=idempotency_learner,
                        raw_key=idempotency_key,
                        fingerprint=fingerprint,
                        resource_id=material_id,
                    )
                    if (
                        replay.status == "completed"
                        and replay.resource_id == material_id
                        and replay.response == response_payload
                    ):
                        return response_payload
                elif existing_material:
                    return response_payload

            if idempotency_key and reservation_owned:
                if not lock_idempotency_attempt(
                    persist_conn,
                    scope="material",
                    learner_id=idempotency_learner,
                    raw_key=idempotency_key,
                    resource_id=material_id,
                    attempt_token=reservation_attempt_token,
                ):
                    raise RuntimeError("material idempotency reservation was lost")
                if file_content is not None and source_path is None:
                    # Hold the exact-key row lock through storage publication so
                    # an owner fenced out during provider work cannot orphan a
                    # duplicate upload object. Retain the published path across
                    # a fresh-transaction retry so the same request does not
                    # publish a second UUID-keyed object.
                    source_path = storage.save_bytes(file_content, file_name)
                if reservation_reclaimed:
                    # Cleanup and replacement are protected by the same row lock
                    # and commit as publication of this fenced attempt.
                    execute_modify(
                        persist_conn,
                        "DELETE FROM material_chunks WHERE material_id = ?",
                        (material_id,),
                    )
                    execute_modify(
                        persist_conn,
                        "DELETE FROM concepts WHERE material_id = ?",
                        (material_id,),
                    )

            upsert(
                persist_conn,
                "materials",
                {
                    "id": material_id,
                    "subject_tag": subject_tag,
                    "raw_text": raw_text,
                    "source_type": source_type,
                    "source_path": source_path,
                    "created_at": created_at,
                    "knowledge_level": normalized_level,
                },
            )

            for concept, emb in zip(concepts, concept_embeddings):
                upsert(
                    persist_conn,
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

            for chunk_record in chunk_records:
                upsert(
                    persist_conn,
                    "material_chunks",
                    {
                        "id": chunk_record["id"],
                        "material_id": material_id,
                        "chunk_index": chunk_record["chunk_index"],
                        "text": chunk_record["text"],
                        "embedding_json": chunk_record["embedding_json"],
                        "created_at": created_at,
                    },
                )
            if learner_id:
                reel_service.learner_progress(
                    persist_conn,
                    material_id,
                    learner_id,
                )

            if quota_account_id:
                settle_operation(
                    persist_conn,
                    account_id=quota_account_id,
                    operation_key=quota_operation_key,
                    usable_result=bool(concepts),
                )

            if idempotency_key and reservation_owned:
                if not complete_idempotency_key(
                    persist_conn,
                    scope="material",
                    learner_id=idempotency_learner,
                    raw_key=idempotency_key,
                    resource_id=material_id,
                    attempt_token=reservation_attempt_token,
                    response=response_payload,
                ):
                    raise RuntimeError("material idempotency reservation was lost")
            publication_body_completed = True
            return response_payload

        _run_generation_db_transaction(
            "material_final_persistence",
            persist_material,
            replay_after_unknown_commit=True,
        )
        return response_payload
    except asyncio.CancelledError:
        if quota_account_id:
            _run_generation_db_transaction(
                "material_quota_cancel",
                lambda quota_conn: settle_operation(
                    quota_conn,
                    account_id=quota_account_id,
                    operation_key=quota_operation_key,
                    usable_result=False,
                ),
                replay_after_unknown_commit=True,
            )
        if idempotency_key and reservation_owned:
            _run_generation_db_transaction(
                "material_idempotency_cancel",
                lambda idempotency_conn: release_idempotency_key(
                    idempotency_conn,
                    scope="material",
                    learner_id=idempotency_learner,
                    raw_key=idempotency_key,
                    resource_id=material_id,
                    attempt_token=reservation_attempt_token,
                ),
                replay_after_unknown_commit=True,
            )
        raise
    except Exception:
        if quota_account_id:
            _run_generation_db_transaction(
                "material_quota_failure",
                lambda quota_conn: settle_operation(
                    quota_conn,
                    account_id=quota_account_id,
                    operation_key=quota_operation_key,
                    usable_result=False,
                ),
                replay_after_unknown_commit=True,
            )
        if idempotency_key and reservation_owned:
            _run_generation_db_transaction(
                "material_idempotency_failure",
                lambda idempotency_conn: release_idempotency_key(
                    idempotency_conn,
                    scope="material",
                    learner_id=idempotency_learner,
                    raw_key=idempotency_key,
                    resource_id=material_id,
                    attempt_token=reservation_attempt_token,
                ),
                replay_after_unknown_commit=True,
            )
        raise


@app.patch("/api/materials/{material_id}/level")
def update_material_level(material_id: str, request: Request, payload: MaterialLevelUpdateRequest):
    from .services.knowledge_level import effective_level_target
    with get_conn(transactional=True) as conn:
        row = fetch_one(conn, "SELECT id FROM materials WHERE id = ? LIMIT 1", (material_id,))
        if not row:
            raise HTTPException(status_code=404, detail="material not found")
        learner_id = _resolve_learner_identity(conn, request, required=False)
        if learner_id:
            _lock_learner_adaptation(
                conn,
                material_id=material_id,
                learner_id=learner_id,
            )
            reel_service.set_learner_level(conn, material_id, learner_id, payload.knowledge_level)
            _cancel_stale_active_adaptation_jobs(
                conn,
                material_id=material_id,
                learner_id=learner_id,
                adaptation_fingerprint=_learner_adaptation_fingerprint(
                    conn,
                    material_id=material_id,
                    learner_id=learner_id,
                ),
            )
        else:
            # Compatibility for pre-personalization clients: their PATCH request
            # carried no identity, so retain the old material-default behavior.
            execute_modify(
                conn,
                "UPDATE materials SET knowledge_level = ?, level_adjustment = 0.0 WHERE id = ?",
                (payload.knowledge_level, material_id),
            )
    return {
        "knowledge_level": payload.knowledge_level,
        "effective_level_target": effective_level_target(payload.knowledge_level, 0.0),
    }


@app.post(
    "/api/reels/generate",
    response_model=ReelsGenerateResponse,
    responses={202: {"model": GenerationJobQueuedResponse}},
)
async def generate_reels(request: Request, payload: ReelsGenerateRequest):
    _require_community_client_identity(request)
    _reject_multi_platform_search(payload.multi_platform_search)
    min_relevance = _normalize_min_relevance(payload.min_relevance)
    safe_video_duration_pref = _normalize_preferred_video_duration(payload.preferred_video_duration)
    excluded_video_ids = _normalize_excluded_video_ids(payload.exclude_video_ids)
    continuation_token = str(payload.continuation_token or "").strip() or None
    retry_terminal_job_id = str(payload.retry_terminal_job_id or "").strip() or None
    if continuation_token and retry_terminal_job_id:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "invalid_terminal_retry",
                "message": "A terminal retry cannot also continue the exhausted batch.",
            },
        )
    requested_num_reels = max(
        1,
        min(
            GENERATION_OUTPUT_CEILINGS[payload.generation_mode],
            int(payload.num_reels),
        ),
    )
    submission_job_id = str(uuid.uuid4())
    generation_submit_rate_checked = False

    def submit_generation(conn: Any) -> tuple[str, dict[str, Any]]:
        nonlocal generation_submit_rate_checked
        material = fetch_one(
            conn,
            "SELECT id FROM materials WHERE id = ?",
            (payload.material_id,),
        )
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")
        learner_id = _resolve_learner_identity(conn, request)
        committed_submission = get_generation_job(conn, submission_job_id)
        if committed_submission is not None:
            if (
                str(committed_submission.get("material_id") or "")
                != payload.material_id
                or str(committed_submission.get("learner_id") or "")
                != learner_id
            ):
                raise RuntimeError("generation submission identity collision")
            return "job", committed_submission
        _lock_learner_adaptation(
            conn,
            material_id=payload.material_id,
            learner_id=learner_id,
        )
        learner_progress = reel_service.learner_progress(conn, payload.material_id, learner_id)
        learner_knowledge_level = str(learner_progress.get("selected_level") or "beginner")
        try:
            learner_difficulty_target = effective_level_target(
                learner_knowledge_level,
                learner_progress.get("global_adjustment"),
            )
        except (TypeError, ValueError, OverflowError):
            learner_knowledge_level = "beginner"
            learner_difficulty_target = effective_level_target("beginner", 0.0)
        adaptation_fingerprint = _learner_adaptation_fingerprint(
            conn,
            material_id=payload.material_id,
            learner_id=learner_id,
        )
        content_fingerprint = material_content_fingerprint(
            conn,
            payload.material_id,
            payload.concept_id,
        )
        request_key = build_durable_request_key(
            material_id=payload.material_id,
            concept_id=payload.concept_id,
            content_fingerprint=content_fingerprint,
            learner_id=learner_id,
            knowledge_level=learner_knowledge_level,
            generation_mode=payload.generation_mode,
            creative_commons_only=payload.creative_commons_only,
            source_duration=safe_video_duration_pref,
            target_clip_duration_sec=0,
            target_clip_duration_min_sec=None,
            target_clip_duration_max_sec=None,
            min_relevance=min_relevance,
            exclude_video_ids=excluded_video_ids,
            continuation_token=continuation_token,
            adaptation_fingerprint=adaptation_fingerprint,
        )
        request_params = {
            "material_id": payload.material_id,
            "concept_id": payload.concept_id,
            "num_reels": requested_num_reels,
            "exclude_video_ids": excluded_video_ids,
            "continuation_token": continuation_token,
            "creative_commons_only": payload.creative_commons_only,
            "generation_mode": payload.generation_mode,
            "min_relevance": min_relevance,
            "preferred_video_duration": safe_video_duration_pref,
            "knowledge_level": learner_knowledge_level,
            "effective_level_target": learner_difficulty_target,
            "adaptation_fingerprint": adaptation_fingerprint,
            "language": "en",
        }
        _cancel_stale_active_adaptation_jobs(
            conn,
            material_id=payload.material_id,
            learner_id=learner_id,
            adaptation_fingerprint=adaptation_fingerprint,
        )
        terminal_retry_job: dict[str, Any] | None = None
        terminal_retry_active_job: dict[str, Any] | None = None
        if retry_terminal_job_id:
            terminal_retry_job = get_generation_job(conn, retry_terminal_job_id)
            retry_matches = bool(
                terminal_retry_job
                and str(terminal_retry_job.get("material_id") or "")
                == payload.material_id
                and (str(terminal_retry_job.get("concept_id") or "") or None)
                == payload.concept_id
                and str(terminal_retry_job.get("learner_id") or "") == learner_id
                and str(terminal_retry_job.get("content_fingerprint") or "")
                == content_fingerprint
                and _generation_job_matches_request_params(
                    terminal_retry_job,
                    request_params,
                )
                and _generation_job_allows_terminal_retry(conn, terminal_retry_job)
            )
            if not retry_matches:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "invalid_terminal_retry",
                        "message": (
                            "Only the current empty retryable reel batch can be retried."
                        ),
                    },
                )
            latest_retry_job = _latest_compatible_generation_job(
                conn,
                material_id=payload.material_id,
                learner_id=learner_id,
                concept_id=payload.concept_id,
                content_fingerprint=content_fingerprint,
                request_params=request_params,
            )
            latest_retry_job_id = str(
                (latest_retry_job or {}).get("id") or ""
            ).strip()
            if latest_retry_job_id != retry_terminal_job_id:
                latest_retry_status = str(
                    (latest_retry_job or {}).get("status") or ""
                ).strip()
                if (
                    latest_retry_status in {"queued", "running"}
                    and str((latest_retry_job or {}).get("request_key") or "")
                    == request_key
                ):
                    terminal_retry_active_job = latest_retry_job
                else:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "invalid_terminal_retry",
                            "message": (
                                "A newer reel batch has replaced the terminal batch."
                            ),
                        },
                    )
        source_generation_id: str | None = None
        continuation_job: dict[str, Any] | None = None
        fresh_root_recovery = False
        fresh_retry_source_budget = False
        if continuation_token:
            continuation_job = get_generation_job(conn, continuation_token)
            continuation_status = str((continuation_job or {}).get("status") or "")
            continuation_has_source_work = bool(
                continuation_status in {"failed", "exhausted"}
                and _generation_job_has_retryable_source_work(
                    conn,
                    continuation_job,
                )
            )
            continuation_params = _job_request_params(continuation_job or {})
            continuation_matches = bool(
                continuation_job
                and str(continuation_job.get("material_id") or "") == payload.material_id
                and (str(continuation_job.get("concept_id") or "") or None) == payload.concept_id
                and str(continuation_job.get("learner_id") or "") == learner_id
                and str(continuation_job.get("content_fingerprint") or "") == content_fingerprint
                and str(continuation_params.get("knowledge_level") or "beginner")
                == learner_knowledge_level
                and str(continuation_params.get("generation_mode") or "slow")
                == payload.generation_mode
                and bool(continuation_params.get("creative_commons_only"))
                == bool(payload.creative_commons_only)
                and _normalize_preferred_video_duration(
                    str(continuation_params.get("preferred_video_duration") or "any")
                ) == safe_video_duration_pref
                and _normalize_min_relevance(continuation_params.get("min_relevance"))
                == min_relevance
                and sorted(_normalize_excluded_video_ids(
                    continuation_params.get("exclude_video_ids") or []
                )) == sorted(excluded_video_ids)
                and str(continuation_params.get("request_schema_version") or "")
                == GENERATION_REQUEST_SCHEMA_VERSION
                and str(continuation_params.get("adaptation_fingerprint") or "")
                == adaptation_fingerprint
            )
            if not continuation_matches or (
                continuation_status not in {"completed", "partial"}
                and not continuation_has_source_work
                and continuation_status != "exhausted"
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "invalid_continuation_token",
                        "message": "The requested reel batch can no longer be continued.",
                    },
                )
            if continuation_status == "exhausted" and not continuation_has_source_work:
                return "response", {
                    "reels": [],
                    "generation_id": None,
                    "response_profile": "unified",
                    "batch_id": continuation_token,
                    "batch_size": 0,
                    "continuation_token": continuation_token,
                    "terminal_status": "exhausted",
                }
            source_generation_id = (
                str(continuation_job.get("result_generation_id") or "").strip()
                or None
            )
            if source_generation_id is None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "invalid_continuation_token",
                        "message": "The previous reel batch has no reusable generation state.",
                    },
                )
        completed_job = find_completed_generation_job(conn, request_key)
        active_job = find_active_generation_job(conn, request_key)
        if (
            continuation_job is not None
            and completed_job is None
            and active_job is None
        ):
            exact_terminal = _latest_exact_generation_job(
                conn,
                material_id=payload.material_id,
                learner_id=learner_id,
                request_key=request_key,
            )
            terminal_retry_detail = _failed_global_provider_terminal_retry_detail(
                conn,
                exact_terminal,
            )
            if terminal_retry_detail is not None:
                raise HTTPException(status_code=409, detail=terminal_retry_detail)
            if str((exact_terminal or {}).get("status") or "") == "failed":
                if _generation_job_has_retryable_source_work(
                    conn,
                    exact_terminal,
                ):
                    source_generation_id = str(
                        exact_terminal.get("result_generation_id") or ""
                    ).strip()
                elif _generation_job_has_failed_source_attempts(
                    conn,
                    exact_terminal,
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "source_retry_exhausted",
                            "message": (
                                "The failed video sources exhausted their bounded "
                                "analysis retries."
                            ),
                        },
                    )
        if terminal_retry_job is not None:
            active_job = terminal_retry_active_job
            completed_job = None
            if active_job is None:
                if _generation_job_has_retryable_source_work(
                    conn,
                    terminal_retry_job,
                ):
                    source_generation_id = str(
                        terminal_retry_job.get("result_generation_id") or ""
                    ).strip() or None
                    fresh_retry_source_budget = source_generation_id is not None
                else:
                    if (
                        str(terminal_retry_job.get("terminal_error_code") or "")
                        in JOB_GLOBAL_PROVIDER_ERROR_CODES
                    ):
                        source_generation_id = str(
                            terminal_retry_job.get("source_generation_id") or ""
                        ).strip() or None
                        fresh_retry_source_budget = source_generation_id is not None
                    fresh_root_recovery = True
        elif continuation_job is None:
            latest_compatible_job = _latest_compatible_generation_job(
                conn,
                material_id=payload.material_id,
                learner_id=learner_id,
                concept_id=payload.concept_id,
                content_fingerprint=content_fingerprint,
                request_params=request_params,
            )
            latest_status = str(
                (latest_compatible_job or {}).get("status") or ""
            )
            if latest_status in {"queued", "running"}:
                active_job = latest_compatible_job
                completed_job = None
            elif latest_status in {"completed", "partial"}:
                completed_job = latest_compatible_job
                active_job = None
            elif latest_status == "exhausted":
                if _generation_job_has_retryable_source_work(
                    conn,
                    latest_compatible_job,
                ):
                    completed_job = None
                    active_job = None
                    source_generation_id = str(
                        latest_compatible_job.get("result_generation_id") or ""
                    ).strip() or None
                    fresh_retry_source_budget = source_generation_id is not None
                else:
                    completed_job = latest_compatible_job
                    active_job = None
            elif latest_status == "failed":
                terminal_retry_detail = (
                    _failed_global_provider_terminal_retry_detail(
                        conn,
                        latest_compatible_job,
                    )
                )
                if terminal_retry_detail is not None:
                    raise HTTPException(status_code=409, detail=terminal_retry_detail)
                if _generation_job_has_retryable_source_work(
                    conn,
                    latest_compatible_job,
                ):
                    completed_job = None
                    active_job = None
                    source_generation_id = str(
                        latest_compatible_job.get("result_generation_id") or ""
                    ).strip() or None
                    fresh_retry_source_budget = source_generation_id is not None
                elif _generation_job_has_failed_source_attempts(
                    conn,
                    latest_compatible_job,
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "source_retry_exhausted",
                            "message": (
                                "The failed video sources exhausted their bounded "
                                "analysis retries."
                            ),
                        },
                    )
        if completed_job:
            cached_reels = _generation_job_reels(
                conn,
                completed_job,
                requested_override=requested_num_reels,
            )
            return "response", {
                "reels": cached_reels,
                "generation_id": str(completed_job.get("result_generation_id") or "") or None,
                "response_profile": "unified",
                "batch_id": str(completed_job.get("id") or "") or None,
                "batch_size": len(cached_reels),
                "continuation_token": str(completed_job.get("id") or "") or None,
                "terminal_status": str(completed_job.get("status") or "completed"),
                "reconciliation_tail_reel_ids": (
                    _stored_generation_reconciliation_tail_ids(
                        conn,
                        str(
                            completed_job.get("result_generation_id") or ""
                        ).strip(),
                    )
                ),
            }
        elif (
            not active_job
            and continuation_job is None
            and source_generation_id is None
            and not fresh_root_recovery
        ):
            source_generation_id = _verified_cross_request_source_generation(
                conn,
                material_id=payload.material_id,
                learner_id=learner_id,
                request_key=request_key,
                concept_id=payload.concept_id,
                content_fingerprint=content_fingerprint,
                request_params=request_params,
            )
            if (
                source_generation_id
                and _generation_chain_meets_source_budget(
                    conn,
                    generation_id=source_generation_id,
                    generation_mode=payload.generation_mode,
                )
            ):
                request_params["fresh_source_budget"] = True
        if fresh_retry_source_budget:
            request_params["fresh_source_budget"] = True
        quota_account: dict[str, str] = {}
        quota_operation_key = f"material:{payload.material_id}"

        def before_generation_create() -> None:
            nonlocal generation_submit_rate_checked
            if not generation_submit_rate_checked:
                _enforce_rate_limit(
                    request,
                    "generation-submit",
                    limit=REELS_GENERATE_RATE_LIMIT_PER_WINDOW,
                )
                generation_submit_rate_checked = True
            account = _require_verified_provider_account(conn, request)
            account_id = str(account["id"])
            quota_account["id"] = account_id

        job_row, created = _submit_bounded_generation_job(
            conn,
            material_id=payload.material_id,
            concept_id=payload.concept_id,
            request_key=request_key,
            content_fingerprint=content_fingerprint,
            learner_id=learner_id,
            request_params=request_params,
            source_generation_id=source_generation_id,
            job_id=submission_job_id,
            before_create=before_generation_create,
        )
        if created and quota_account.get("id"):
            attach_reservation_to_job(
                conn,
                account_id=quota_account["id"],
                operation_key=quota_operation_key,
                generation_job_id=str(job_row["id"]),
                material_id=payload.material_id,
            )
        return "job", job_row

    submission_kind, submission = _run_generation_db_transaction(
        "api_generate_submit",
        submit_generation,
        replay_after_unknown_commit=True,
    )
    if submission_kind == "response":
        return submission
    job_row = submission
    _wake_generation_worker()
    job_id = str(job_row.get("id") or "")
    status = str(job_row.get("status") or "queued")
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": status,
            "status_url": f"/api/reels/generation-status/{job_id}",
            "stream_url": f"/api/reels/generation-stream/{job_id}",
        },
    )


def _ingest_error_to_http(exc: IngestError) -> HTTPException:
    """Map an IngestError to the appropriate HTTPException. Shared by both ingest endpoints."""
    status = int(getattr(exc, "status_code", 500) or 500)
    detail_payload: dict[str, Any] = {
        "error": exc.__class__.__name__,
        "message": exc.message,
    }
    if exc.detail:
        detail_payload["detail"] = exc.detail
    headers: dict[str, str] = {}
    if isinstance(exc, IngestRateLimitedError):
        retry = max(1, int(round(exc.retry_after_sec)))
        headers["Retry-After"] = str(retry)
    http_exc = HTTPException(status_code=status, detail=detail_payload)
    if headers:
        http_exc.headers = headers
    return http_exc


@app.get(
    "/api/reels/generation-status/{job_id}",
    response_model=GenerationJobStatusResponse,
)
def generation_status(request: Request, job_id: str):
    _require_community_client_identity(request)
    _enforce_rate_limit(
        request,
        "reels-generation-status",
        limit=GENERATION_JOB_STATUS_RATE_LIMIT_PER_WINDOW,
    )
    clean_job_id = str(job_id or "").strip()
    with get_conn(transactional=True) as conn:
        learner_id = _resolve_learner_identity(conn, request)
        job_row = get_generation_job(conn, clean_job_id)
        if (
            not job_row
            or str(job_row.get("learner_id") or LEGACY_LEARNER_ID) != learner_id
        ):
            raise HTTPException(status_code=404, detail="job_id not found")
        expire_stale_generation_job(conn, job_id=clean_job_id)
        job_row = get_generation_job(conn, clean_job_id)
        if (
            not job_row
            or str(job_row.get("learner_id") or LEGACY_LEARNER_ID) != learner_id
        ):
            raise HTTPException(status_code=404, detail="job_id not found")
        return _generation_job_status_payload(conn, job_row)


@app.get("/api/reels/generation-stream/{job_id}")
async def generation_stream(request: Request, job_id: str, after_seq: int = 0):
    _require_community_client_identity(request)
    _enforce_rate_limit(
        request,
        "reels-generation-stream",
        limit=GENERATION_JOB_STATUS_RATE_LIMIT_PER_WINDOW,
    )
    clean_job_id = str(job_id or "").strip()
    with get_conn(transactional=True) as conn:
        learner_id = _resolve_learner_identity(conn, request)
        job_row = get_generation_job(conn, clean_job_id)
        if (
            not job_row
            or str(job_row.get("learner_id") or LEGACY_LEARNER_ID) != learner_id
        ):
            raise HTTPException(status_code=404, detail="job_id not found")
        expire_stale_generation_job(conn, job_id=clean_job_id)
        job_row = get_generation_job(conn, clean_job_id)
        if (
            not job_row
            or str(job_row.get("learner_id") or LEGACY_LEARNER_ID) != learner_id
        ):
            raise HTTPException(status_code=404, detail="job_id not found")

    async def ndjson_events():
        cursor = max(0, int(after_seq or 0))
        while True:
            with get_conn(transactional=True) as conn:
                expire_stale_generation_job(conn, job_id=clean_job_id)
                replayed_events = replay_generation_events(
                    conn,
                    job_id=clean_job_id,
                    after_seq=cursor,
                )
                job_row = get_generation_job(conn, clean_job_id)
                if replayed_events:
                    cursor = max(
                        cursor,
                        max(int(event.get("seq") or 0) for event in replayed_events),
                    )
                events = _sanitize_generation_replay_events(
                    conn,
                    job_row,
                    replayed_events,
                )
            for event in events:
                yield f"{json.dumps(event, default=str, separators=(',', ':'))}\n"
            if any(
                str(event.get("type") or "") == "terminal"
                for event in replayed_events
            ):
                return
            if not job_row or (
                str(job_row.get("status") or "") in GENERATION_TERMINAL_STATUSES
                and not events
            ):
                return
            if await request.is_disconnected():
                # Disconnects only unsubscribe. The leased worker keeps running.
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(
        ndjson_events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/api/reels/generation-jobs/{job_id}/cancel",
    response_model=GenerationJobStatusResponse,
)
def cancel_generation_job(request: Request, job_id: str):
    _require_community_client_identity(request)
    _enforce_rate_limit(
        request,
        "reels-generation-cancel",
        limit=GENERATION_JOB_STATUS_RATE_LIMIT_PER_WINDOW,
    )
    with get_conn(transactional=True) as conn:
        clean_job_id = str(job_id or "").strip()
        learner_id = _resolve_learner_identity(conn, request)
        existing_job = get_generation_job(conn, clean_job_id)
        if (
            not existing_job
            or str(existing_job.get("learner_id") or LEGACY_LEARNER_ID)
            != learner_id
        ):
            raise HTTPException(status_code=404, detail="job_id not found")
        job_row = request_generation_cancellation(
            conn,
            job_id=clean_job_id,
        )
        if not job_row:
            raise HTTPException(status_code=404, detail="job_id not found")
        return _generation_job_status_payload(conn, job_row)


async def _run_disconnect_cancellable(
    request: Request,
    work: Callable[[Callable[[], bool]], Any],
) -> Any:
    """Run the synchronous pipeline in a worker while observing disconnects."""
    cancel_event = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(work, cancel_event.is_set))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.05)
            if task in done:
                return await task
            if await request.is_disconnected():
                cancel_event.set()
    except asyncio.CancelledError:
        cancel_event.set()
        raise


def _run_adaptive_mutation_transaction(
    work: Callable[[Any], Any],
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> Any:
    """Run one adaptive mutation with one fresh-transaction PostgreSQL retry."""
    for attempt in range(2):
        if should_cancel is not None and should_cancel():
            raise AssessmentCancelledError("Assessment request cancelled.")
        try:
            with get_conn(transactional=True) as conn:
                return work(conn)
        except Exception as exc:
            if (
                isinstance(
                    exc,
                    (
                        AssessmentCancelledError,
                        DatabaseIntegrityError,
                        HTTPException,
                        ValueError,
                    ),
                )
                or attempt > 0
                or not is_transient_postgres_transaction_error(exc)
            ):
                raise
            if should_cancel is not None and should_cancel():
                raise AssessmentCancelledError(
                    "Assessment request cancelled."
                ) from exc
            logger.warning(
                "adaptive mutation transaction failed transiently; retrying once: %s",
                str(exc).splitlines()[0][:180],
            )
    raise RuntimeError("adaptive mutation retry loop exhausted")


def _canonical_youtube_source_or_422(
    source_url: str,
    *,
    allowed_kinds: set[str],
) -> str:
    canonical = canonicalize_youtube_url(source_url, allowed_kinds=allowed_kinds)
    if canonical is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_youtube_url",
                "message": (
                    "A canonical YouTube "
                    + "/".join(sorted(allowed_kinds))
                    + " URL is required. Instagram and TikTok are unsupported."
                ),
            },
        )
    return str(canonical["canonical_url"])


def _reject_multi_platform_search(enabled: bool) -> None:
    if enabled:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_retrieval_platform",
                "message": "multi_platform_search=true is unsupported; retrieval is YouTube-only.",
            },
        )


@app.post("/api/ingest/url", response_model=IngestResult)
async def ingest_url_endpoint(request: Request, payload: IngestRequest) -> IngestResult:
    """
    Process one canonical YouTube video URL with a timestamped transcript.
    """
    _enforce_rate_limit(request, "ingest-url", limit=INGEST_URL_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(payload.multi_platform_search)
    canonical_source_url = _canonical_youtube_source_or_422(
        payload.source_url,
        allowed_kinds={"video"},
    )
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Reel ingestion is disabled in serverless mode.",
            },
        )

    quota = _begin_sync_search_quota(
        request,
        surface="ingest-url",
        request_fingerprint=build_idempotency_fingerprint(
            {
                "contract": "ingest_url_v1",
                **payload.model_dump(),
                "source_url": canonical_source_url,
            }
        ),
        material_id=payload.material_id,
    )
    if quota and quota.get("replay_response") is not None:
        return IngestResult.model_validate(quota["replay_response"])
    try:
        result = await _run_disconnect_cancellable(
            request,
            lambda should_cancel: ingestion_pipeline.ingest_url(
                source_url=canonical_source_url,
                material_id=payload.material_id,
                concept_id=payload.concept_id,
                target_clip_duration_sec=payload.target_clip_duration_sec,
                target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
                target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
                language=payload.language,
                should_cancel=should_cancel,
            ),
        )
    except ClipEngineCancellationError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestTranscriptionError,
        IngestSegmentationError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.warning(
            "ingest_url failed: %s",
            json.dumps(
                {
                    "source_url": payload.source_url,
                    "error": exc.__class__.__name__,
                    "message": exc.message,
                },
                sort_keys=True,
            ),
        )
        raise _ingest_error_to_http(exc) from exc
    except IngestError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.exception("ingest_url crashed for %s", payload.source_url)
        raise _ingest_error_to_http(exc) from exc
    except asyncio.CancelledError:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    except Exception:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    _settle_sync_search_quota(
        quota,
        usable_result=True,
        response=result.model_dump(mode="json"),
    )
    return result


@app.post("/api/ingest/topic-cut", response_model=IngestTopicCutResult)
async def ingest_topic_cut_endpoint(request: Request, payload: IngestTopicCutRequest) -> IngestTopicCutResult:
    """
    Topic-aware variant of `/api/ingest/url`. Same download + transcribe path,
    but instead of returning ONE clip per video this endpoint returns one reel
    per topic the creator introduces (Shorts are returned with `reels: []` and
    `is_short: true` so the caller can leave them untouched).

    Each entry in `reels` decodes cleanly into the iOS `Reel` struct via the
    existing decoder — no client schema changes are required.
    """
    _enforce_rate_limit(request, "ingest-topic-cut", limit=INGEST_TOPIC_CUT_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(payload.multi_platform_search)
    canonical_source_url = _canonical_youtube_source_or_422(
        payload.source_url,
        allowed_kinds={"video"},
    )
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Topic-cut ingestion is disabled in serverless mode.",
            },
        )

    quota = _begin_sync_search_quota(
        request,
        surface="ingest-topic-cut",
        request_fingerprint=build_idempotency_fingerprint(
            {
                "contract": "ingest_topic_cut_v1",
                **payload.model_dump(),
                "source_url": canonical_source_url,
            }
        ),
        material_id=payload.material_id,
    )
    if quota and quota.get("replay_response") is not None:
        return IngestTopicCutResult.model_validate(quota["replay_response"])
    try:
        result = await _run_disconnect_cancellable(
            request,
            lambda should_cancel: ingestion_pipeline.ingest_topic_cut(
                source_url=canonical_source_url,
                material_id=payload.material_id,
                concept_id=payload.concept_id,
                language=payload.language,
                use_llm=payload.use_llm,
                query=payload.query,
                should_cancel=should_cancel,
            ),
        )
    except ClipEngineCancellationError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestTranscriptionError,
        IngestSegmentationError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.warning(
            "ingest_topic_cut failed: %s",
            json.dumps(
                {
                    "source_url": payload.source_url,
                    "error": exc.__class__.__name__,
                    "message": exc.message,
                },
                sort_keys=True,
            ),
        )
        raise _ingest_error_to_http(exc) from exc
    except IngestError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.exception("ingest_topic_cut crashed for %s", payload.source_url)
        raise _ingest_error_to_http(exc) from exc
    except asyncio.CancelledError:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    except Exception:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    _settle_sync_search_quota(
        quota,
        usable_result=result.reel_count > 0,
        response=result.model_dump(mode="json"),
    )
    return result


@app.post("/api/ingest/search", response_model=IngestSearchResult)
async def ingest_search_endpoint(request: Request, payload: IngestSearchRequest) -> IngestSearchResult:
    """
    YouTube-only discovery with timestamped-transcript ingestion.
    """
    _enforce_rate_limit(request, "ingest-search", limit=INGEST_SEARCH_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(payload.multi_platform_search)
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Reel ingestion is disabled in serverless mode.",
            },
        )

    quota = _begin_sync_search_quota(
        request,
        surface="ingest-search",
        request_fingerprint=build_idempotency_fingerprint(
            {"contract": "ingest_search_v1", **payload.model_dump()}
        ),
        material_id=payload.material_id,
        charge_search=not bool(payload.exclude_video_ids),
    )
    if quota and quota.get("replay_response") is not None:
        return IngestSearchResult.model_validate(quota["replay_response"])
    try:
        result = await _run_disconnect_cancellable(
            request,
            lambda should_cancel: ingestion_pipeline.ingest_search(
                query=payload.query,
                platforms=payload.platforms,
                max_per_platform=payload.max_per_platform,
                material_id=payload.material_id,
                concept_id=payload.concept_id,
                target_clip_duration_sec=payload.target_clip_duration_sec,
                target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
                target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
                language=payload.language,
                exclude_video_ids=payload.exclude_video_ids,
                should_cancel=should_cancel,
            ),
        )
    except ClipEngineCancellationError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.warning(
            "ingest_search failed: %s",
            json.dumps(
                {
                    "query": payload.query,
                    "platforms": payload.platforms,
                    "error": exc.__class__.__name__,
                    "message": exc.message,
                },
                sort_keys=True,
            ),
        )
        raise _ingest_error_to_http(exc) from exc
    except IngestError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.exception("ingest_search crashed for query=%s", payload.query)
        raise _ingest_error_to_http(exc) from exc
    except asyncio.CancelledError:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    except Exception:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    _settle_sync_search_quota(
        quota,
        usable_result=result.succeeded > 0,
        response=result.model_dump(mode="json"),
    )
    return result


@app.post("/api/ingest/feed", response_model=IngestFeedResult)
async def ingest_feed_endpoint(request: Request, payload: IngestFeedRequest) -> IngestFeedResult:
    """
    Resolve a profile / hashtag / playlist URL to a bounded list of individual reel URLs
    and ingest each in a small thread pool. Per-item failures are recorded in the response
    (`items[*].status == "error"`) rather than aborting the whole call.
    """
    _enforce_rate_limit(request, "ingest-feed", limit=INGEST_FEED_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(payload.multi_platform_search)
    canonical_feed_url = _canonical_youtube_source_or_422(
        payload.feed_url,
        allowed_kinds={"video", "playlist", "channel"},
    )
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Reel ingestion is disabled in serverless mode.",
            },
        )

    quota = _begin_sync_search_quota(
        request,
        surface="ingest-feed",
        request_fingerprint=build_idempotency_fingerprint(
            {
                "contract": "ingest_feed_v1",
                **payload.model_dump(),
                "feed_url": canonical_feed_url,
            }
        ),
        material_id=payload.material_id,
    )
    if quota and quota.get("replay_response") is not None:
        return IngestFeedResult.model_validate(quota["replay_response"])
    try:
        result = await _run_disconnect_cancellable(
            request,
            lambda should_cancel: ingestion_pipeline.ingest_feed(
                feed_url=canonical_feed_url,
                max_items=payload.max_items,
                material_id=payload.material_id,
                concept_id=payload.concept_id,
                target_clip_duration_sec=payload.target_clip_duration_sec,
                target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
                target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
                language=payload.language,
                should_cancel=should_cancel,
            ),
        )
    except ClipEngineCancellationError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.warning(
            "ingest_feed failed: %s",
            json.dumps(
                {
                    "feed_url": payload.feed_url,
                    "error": exc.__class__.__name__,
                    "message": exc.message,
                },
                sort_keys=True,
            ),
        )
        raise _ingest_error_to_http(exc) from exc
    except IngestError as exc:
        _settle_sync_search_quota(quota, usable_result=False)
        logger.exception("ingest_feed crashed for %s", payload.feed_url)
        raise _ingest_error_to_http(exc) from exc
    except asyncio.CancelledError:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    except Exception:
        _settle_sync_search_quota(quota, usable_result=False)
        raise
    _settle_sync_search_quota(
        quota,
        usable_result=result.succeeded > 0,
        response=result.model_dump(mode="json"),
    )
    return result


def _material_preflight_query(conn, material_id: str, concept_id: str | None = None) -> str:
    material = fetch_one(
        conn,
        "SELECT subject_tag, raw_text FROM materials WHERE id = ?",
        (material_id,),
    )
    if not material:
        raise HTTPException(status_code=404, detail="material_id not found")
    concept = None
    if concept_id:
        concept = fetch_one(
            conn,
            "SELECT title, summary FROM concepts WHERE id = ? AND material_id = ?",
            (concept_id, material_id),
        )
        if not concept:
            raise HTTPException(status_code=404, detail="concept_id not found for material")
    else:
        concept = fetch_one(
            conn,
            "SELECT title, summary FROM concepts WHERE material_id = ? ORDER BY created_at, id LIMIT 1",
            (material_id,),
        )
    candidates = [
        (concept or {}).get("title"),
        material.get("subject_tag"),
        (concept or {}).get("summary"),
        str(material.get("raw_text") or "")[:240],
    ]
    query = " ".join(
        " ".join(str(value or "").split())
        for value in candidates
        if str(value or "").strip()
    )
    if not query.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "blank_retrieval_topic", "message": "Material has no non-blank retrieval topic."},
        )
    return query[:500]


def _provider_error_to_http(exc: ClipEngineProviderError) -> HTTPException:
    status_by_code = {
        "transcript_unavailable": 422,
        "provider_quota_exhausted": 402,
        "provider_rate_limited": 429,
        "provider_configuration": 503,
        "provider_authentication": 503,
    }
    status = status_by_code.get(exc.code, 502)
    headers = None
    if exc.retry_after_sec is not None:
        headers = {"Retry-After": str(max(1, int(math.ceil(exc.retry_after_sec))))}
    return HTTPException(status_code=status, detail=exc.as_dict(), headers=headers)


def _preflight_evidence(
    *,
    query: str,
    filters: dict[str, Any],
    allow_provider: bool,
) -> dict[str, Any]:
    normalized = normalize_provider_filters(filters)
    key = search_cache_key(
        query=query,
        filters=normalized,
        language="en",
        page_token=None,
    )
    cached = DatabaseProviderCache().get_search(key)
    if cached is not None:
        videos = cached.payload.get("videos") or []
        return {
            "availability": "available" if videos else "unavailable",
            "evidence_source": "cache",
            "evidence_age_sec": round(float(cached.age_sec), 3),
            "candidate_count": len(videos),
            "filters_applied": normalized,
            "provider_called": False,
        }
    if not allow_provider:
        return {
            "availability": "unknown",
            "evidence_source": "none",
            "evidence_age_sec": None,
            "candidate_count": 0,
            "filters_applied": normalized,
            "provider_called": False,
        }
    result = supadata_search_one(query, filters, language="en")
    videos = result.get("videos") or []
    return {
        "availability": "available" if videos else "unavailable",
        "evidence_source": "cache" if result.get("cache_hit") else "provider",
        "evidence_age_sec": round(float(result.get("evidence_age_sec") or 0.0), 3),
        "candidate_count": len(videos),
        "filters_applied": result.get("filters_applied") or normalized,
        "provider_called": not bool(result.get("cache_hit")),
    }


@app.post("/api/reels/can-generate", response_model=ReelsCanGenerateResponse)
def can_generate_reels(request: Request, payload: ReelsGenerateRequest):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "reels-can-generate", limit=REELS_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(payload.multi_platform_search)
    filters = {
        "creative_commons_only": payload.creative_commons_only,
        "duration": payload.preferred_video_duration,
    }
    try:
        with get_conn() as conn:
            query = _material_preflight_query(conn, payload.material_id, payload.concept_id)
        evidence = _preflight_evidence(
            query=query,
            filters=filters,
            allow_provider=False,
        )
    except ClipEngineProviderError as exc:
        raise _provider_error_to_http(exc) from exc
    evidence["message"] = (
        "Cached YouTube evidence contains matching candidates."
        if evidence["availability"] == "available"
        else "No matching YouTube candidates were found."
        if evidence["availability"] == "unavailable"
        else "No cached search evidence is available yet."
    )
    evidence.pop("provider_called", None)
    return evidence

@app.post("/api/reels/can-generate-any", response_model=ReelsCanGenerateAnyResponse)
def can_generate_reels_any(request: Request, payload: ReelsCanGenerateAnyRequest):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "reels-can-generate-any", limit=REELS_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(payload.multi_platform_search)
    filters = {
        "creative_commons_only": payload.creative_commons_only,
        "duration": payload.preferred_video_duration,
    }
    with get_conn() as conn:
        requested_ids: list[str] = []
        for raw_id in payload.material_ids:
            clean_id = str(raw_id or "").strip()
            if clean_id and clean_id not in requested_ids:
                requested_ids.append(clean_id)
        if requested_ids:
            material_ids = requested_ids[:5]
        else:
            material_ids = [
                str(row.get("id") or "")
                for row in fetch_all(conn, "SELECT id FROM materials ORDER BY created_at DESC LIMIT 5")
            ]
        queries: list[tuple[str, str]] = []
        for material_id in material_ids:
            try:
                queries.append((material_id, _material_preflight_query(conn, material_id)))
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise

    if not queries:
        return {
            "availability": "unknown",
            "evidence_source": "none",
            "evidence_age_sec": None,
            "candidate_count": 0,
            "filters_applied": normalize_provider_filters(filters),
            "materials_checked": 0,
            "message": "No study materials are available to inspect.",
        }

    cached_results = [
        _preflight_evidence(query=query, filters=filters, allow_provider=False)
        for _material_id, query in queries
    ]
    for result in cached_results:
        if result["availability"] == "available":
            result["materials_checked"] = len(queries)
            result["message"] = "Cached YouTube evidence contains matching candidates."
            result.pop("provider_called", None)
            return result

    known = [result for result in cached_results if result["availability"] != "unknown"]
    all_known = len(known) == len(cached_results)
    evidence_source = "cache" if known else "none"
    ages = [float(result["evidence_age_sec"]) for result in known if result.get("evidence_age_sec") is not None]
    return {
        "availability": "unavailable" if all_known else "unknown",
        "evidence_source": evidence_source,
        "evidence_age_sec": max(ages) if ages else None,
        "candidate_count": max((int(result.get("candidate_count") or 0) for result in known), default=0),
        "filters_applied": normalize_provider_filters(filters),
        "materials_checked": len(queries),
        "message": (
            "Cached evidence found no matching YouTube candidates."
            if all_known
            else "Some materials have no cached search evidence yet."
        ),
    }

@app.get("/api/feed", response_model=FeedResponse)
def feed(
    request: Request,
    material_id: str,
    page: int = 1,
    limit: int = 5,
    autofill: bool = True,
    prefetch: int = INITIAL_READY_REEL_TARGET,
    creative_commons_only: bool = False,
    generation_mode: Literal["slow", "fast"] = "slow",
    min_relevance: float | None = None,
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any",
    target_clip_duration_sec: int | None = None,
    target_clip_duration_min_sec: int | None = None,
    target_clip_duration_max_sec: int | None = None,
    exclude_video_ids: str = "",
    exclude_reel_ids: str = "",
    multi_platform_search: bool = False,
):
    from .services.knowledge_level import effective_level_target as _effective_level_target

    _enforce_rate_limit(request, "feed", limit=FEED_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(multi_platform_search)
    if page < 1:
        page = 1
    # Cap page to bound _ranked_request_reels's cumulative-page loop (O(page))
    # so an anonymous caller cannot exhaust CPU/memory with ?page=1000000.
    if page > 200:
        page = 200
    if limit < 1:
        limit = 1
    if limit > 25:
        limit = 25
    if prefetch < 0:
        prefetch = 0
    if prefetch > 30:
        prefetch = 30

    safe_duration = _normalize_preferred_video_duration(preferred_video_duration)
    # Deprecated clip-duration query fields are accepted for old URLs but inert.
    safe_clip_target, safe_clip_min, safe_clip_max = (0, None, None)
    safe_relevance = _normalize_min_relevance(min_relevance)
    excluded_videos = _parse_excluded_video_ids_param(exclude_video_ids)
    excluded_reels = _parse_excluded_reel_ids_param(exclude_reel_ids)
    feed_submission_job_id = str(uuid.uuid4())
    feed_submit_rate_checked = False
    committed_feed_result: tuple[dict[str, Any], bool] | None = None

    def load_feed(conn: Any) -> tuple[dict[str, Any], bool]:
        nonlocal committed_feed_result, feed_submit_rate_checked
        job_submitted = False
        material = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")
        learner_id = _resolve_learner_identity(conn, request)
        if committed_feed_result is not None and committed_feed_result[1]:
            committed_job = get_generation_job(conn, feed_submission_job_id)
            if (
                committed_job is not None
                and str(committed_job.get("material_id") or "") == material_id
                and str(committed_job.get("learner_id") or "") == learner_id
            ):
                return committed_feed_result
        _lock_learner_adaptation(
            conn,
            material_id=material_id,
            learner_id=learner_id,
        )
        progress = reel_service.learner_progress(conn, material_id, learner_id)
        knowledge_level = str(progress.get("selected_level") or "beginner")
        adaptation_fingerprint = _learner_adaptation_fingerprint(
            conn,
            material_id=material_id,
            learner_id=learner_id,
        )
        try:
            effective_level = _effective_level_target(
                knowledge_level,
                float(progress.get("global_adjustment") or 0.0),
            )
        except ValueError:
            knowledge_level = "beginner"
            effective_level = _effective_level_target("beginner", 0.0)
        content_fingerprint = material_content_fingerprint(conn, material_id)
        request_key = build_durable_request_key(
            material_id=material_id,
            concept_id=None,
            content_fingerprint=content_fingerprint,
            learner_id=learner_id,
            knowledge_level=knowledge_level,
            generation_mode=generation_mode,
            creative_commons_only=creative_commons_only,
            source_duration=safe_duration,
            target_clip_duration_sec=safe_clip_target,
            target_clip_duration_min_sec=safe_clip_min,
            target_clip_duration_max_sec=safe_clip_max,
            min_relevance=safe_relevance,
            exclude_video_ids=excluded_videos,
            adaptation_fingerprint=adaptation_fingerprint,
        )
        request_params = {
            "material_id": material_id,
            "concept_id": None,
            "num_reels": GENERATION_OUTPUT_CEILINGS[generation_mode],
            "exclude_video_ids": excluded_videos,
            "creative_commons_only": creative_commons_only,
            "generation_mode": generation_mode,
            "min_relevance": safe_relevance,
            "preferred_video_duration": safe_duration,
            "knowledge_level": knowledge_level,
            "effective_level_target": effective_level,
            "adaptation_fingerprint": adaptation_fingerprint,
            "language": "en",
        }
        _cancel_stale_active_adaptation_jobs(
            conn,
            material_id=material_id,
            learner_id=learner_id,
            adaptation_fingerprint=adaptation_fingerprint,
        )
        completed_job = find_completed_generation_job(conn, request_key)
        active_job = find_active_generation_job(conn, request_key)
        latest_compatible_job = _latest_compatible_generation_job(
            conn,
            material_id=material_id,
            learner_id=learner_id,
            concept_id=None,
            content_fingerprint=content_fingerprint,
            request_params=request_params,
        )
        latest_status = str((latest_compatible_job or {}).get("status") or "")
        retry_source_job: dict[str, Any] | None = None
        terminal_retry_suppressed = False
        if latest_status in {"queued", "running"}:
            active_job = latest_compatible_job
        elif latest_status in {"completed", "partial"}:
            completed_job = latest_compatible_job
            active_job = None
        elif latest_status == "exhausted":
            if _generation_job_has_retryable_source_work(
                conn,
                latest_compatible_job,
            ):
                completed_job = None
                active_job = None
                retry_source_job = latest_compatible_job
            else:
                completed_job = latest_compatible_job
                active_job = None
        elif latest_status == "failed":
            latest_error_code = str(
                (latest_compatible_job or {}).get("terminal_error_code") or ""
            ).strip()
            if latest_error_code in JOB_GLOBAL_PROVIDER_ERROR_CODES:
                completed_job = None
                active_job = None
                terminal_retry_suppressed = True
            elif _generation_job_has_retryable_source_work(
                conn,
                latest_compatible_job,
            ):
                completed_job = None
                active_job = None
                retry_source_job = latest_compatible_job
            else:
                completed_job = None
                active_job = None
                terminal_retry_suppressed = True
        cross_request_source = False
        cross_request_source_covers_mode = False
        cross_request_source_params: dict[str, Any] = {}
        if retry_source_job:
            generation_id = str(
                retry_source_job.get("result_generation_id") or ""
            ).strip() or None
        elif active_job:
            generation_id = str(
                active_job.get("source_generation_id") or ""
            ).strip() or None
        else:
            generation_id = str(
                (completed_job or {}).get("result_generation_id") or ""
            ).strip() or None
        if generation_id is None and not active_job and not terminal_retry_suppressed:
            head = _fetch_active_generation_row(conn, material_id=material_id, request_key=request_key)
            generation_id = str((head or {}).get("id") or "") or None
        if (
            generation_id is None
            and not completed_job
            and not active_job
            and not terminal_retry_suppressed
        ):
            generation_id = _verified_cross_request_source_generation(
                conn,
                material_id=material_id,
                learner_id=learner_id,
                request_key=request_key,
                concept_id=None,
                content_fingerprint=content_fingerprint,
                request_params=request_params,
                matched_request_params_out=cross_request_source_params,
            )
            cross_request_source = generation_id is not None
            if generation_id is not None:
                cross_request_source_covers_mode = (
                    _generation_chain_meets_source_budget(
                        conn,
                        generation_id=generation_id,
                        generation_mode=generation_mode,
                    )
                )
        ranked = [] if generation_id is None else _current_selection_contract_reels(
            _ranked_request_reels(
                conn,
                material_id=material_id,
                fast_mode=generation_mode == "fast",
                generation_id=generation_id,
                min_relevance=safe_relevance,
                preferred_video_duration=safe_duration,
                target_clip_duration_sec=safe_clip_target,
                target_clip_duration_min_sec=safe_clip_min,
                target_clip_duration_max_sec=safe_clip_max,
                exclude_video_ids=excluded_videos,
                page=page,
                limit=limit,
                learner_id=learner_id,
                exclude_reel_ids=excluded_reels,
                released_only=True,
            )
        )
        legacy_page_start = (page - 1) * limit
        cursor_reel_ids = set(excluded_reels)
        cursor_reel_ids.update(
            _learner_seen_reel_ids(
                conn,
                material_id=material_id,
                learner_id=learner_id,
            )
        )
        removed_cursor_count = _generation_cursor_reel_count(
            conn,
            material_id=material_id,
            generation_id=generation_id,
            reel_ids=cursor_reel_ids,
        )
        page_start = max(
            0,
            legacy_page_start - min(legacy_page_start, removed_cursor_count),
        )
        page_end = page_start + limit
        initial_ready_target = min(
            GENERATION_OUTPUT_CEILINGS[generation_mode],
            max(limit, prefetch),
        )
        initial_reservoir_shortfall = len(ranked) < initial_ready_target
        fresh_cross_request_budget = bool(
            cross_request_source
            and cross_request_source_covers_mode
            and initial_reservoir_shortfall
        )
        cross_request_adaptation_changed = bool(
            cross_request_source
            and str(
                cross_request_source_params.get("adaptation_fingerprint")
                or GENERATION_EMPTY_ADAPTATION_FINGERPRINT
            )
            != adaptation_fingerprint
        )
        if (
            autofill
            and page == 1
            and not terminal_retry_suppressed
            and completed_job is None
            and active_job is None
            and (
                (
                    cross_request_source
                    and (
                        cross_request_adaptation_changed
                        or not cross_request_source_covers_mode
                        or initial_reservoir_shortfall
                    )
                )
                or (not cross_request_source and initial_reservoir_shortfall)
            )
        ):
            target_total = max(1, initial_ready_target)
            try:
                quota_account: dict[str, str] = {}
                quota_operation_key = f"material:{material_id}"

                def before_feed_generation_create() -> None:
                    nonlocal feed_submit_rate_checked
                    if not feed_submit_rate_checked:
                        _enforce_rate_limit(
                            request,
                            "generation-submit",
                            limit=REELS_GENERATE_RATE_LIMIT_PER_WINDOW,
                        )
                        feed_submit_rate_checked = True
                    account = _require_verified_provider_account(conn, request)
                    account_id = str(account["id"])
                    quota_account["id"] = account_id

                active_job, created = _submit_bounded_generation_job(
                    conn,
                    material_id=material_id,
                    concept_id=None,
                    request_key=request_key,
                    content_fingerprint=content_fingerprint,
                    learner_id=learner_id,
                    request_params={
                        **request_params,
                        "num_reels": target_total,
                        "fresh_source_budget": bool(
                            retry_source_job or fresh_cross_request_budget
                        ),
                    },
                    source_generation_id=generation_id,
                    job_id=feed_submission_job_id,
                    before_create=before_feed_generation_create,
                )
                if created and quota_account.get("id"):
                    attach_reservation_to_job(
                        conn,
                        account_id=quota_account["id"],
                        operation_key=quota_operation_key,
                        generation_job_id=str(active_job["id"]),
                        material_id=material_id,
                    )
            except HTTPException as exc:
                if exc.status_code != 429:
                    raise
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                if detail.get("code") == "daily_search_limit_reached":
                    raise
            else:
                job_submitted = True
        reels = ranked[page_start:page_end]
        reported_job = active_job or completed_job or (
            latest_compatible_job if terminal_retry_suppressed else None
        )
        reported_status = str((reported_job or {}).get("status") or "")
        continuation_token = (
            str((reported_job or {}).get("id") or "").strip()
            if reported_status in {"completed", "partial", "exhausted"}
            else str(_job_request_params(reported_job or {}).get("continuation_token") or "").strip()
        ) or None
        response = {
            "page": page,
            "limit": limit,
            "total": len(ranked),
            "reels": reels,
            "generation_id": generation_id,
            "response_profile": "unified" if generation_id else None,
            "generation_job_id": str((reported_job or {}).get("id") or "") or None,
            "generation_job_status": str((reported_job or {}).get("status") or "") or None,
            "continuation_token": continuation_token,
            "effective_generation_mode": generation_mode,
            "generation_mode_overridden": False,
            "knowledge_level": knowledge_level,
            "effective_level_target": effective_level,
        }
        committed_feed_result = (response, job_submitted)
        return committed_feed_result

    response, job_submitted = _run_generation_db_transaction(
        "api_feed_rank_and_submit",
        load_feed,
        replay_after_unknown_commit=True,
    )
    if job_submitted:
        _wake_generation_worker()
    return response

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest):
    _enforce_rate_limit(request, "chat", limit=CHAT_RATE_LIMIT_PER_WINDOW)
    with get_conn(transactional=True) as conn:
        _require_verified_provider_account(conn, request)
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    # Prefer the dedicated chat key. MaterialIntelligenceService retries the
    # primary rotation pool if this credential cannot complete the request.
    chat_gemini_key = (os.environ.get("GEMINI_API_KEY_2") or "").strip() or None
    try:
        answer = await _run_disconnect_cancellable(
            request,
            lambda should_cancel: material_intelligence_service.chat_assistant(
                message=payload.message,
                topic=payload.topic,
                text=payload.text,
                history=history,
                reel_summary=payload.reel_summary,
                video_title=payload.video_title,
                video_description=payload.video_description,
                transcript_snippet=payload.transcript_snippet,
                gemini_api_key_override=chat_gemini_key,
                should_cancel=should_cancel,
            ),
        )
    except ClipEngineCancellationError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except llm_router.TextLLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.as_dict(),
            headers={"Retry-After": "5"},
        ) from exc
    return {"answer": answer}


@app.post("/api/reels/feedback", response_model=FeedbackResponse)
def feedback(request: Request, payload: FeedbackRequest):
    _enforce_rate_limit(request, "feedback", limit=FEEDBACK_RATE_LIMIT_PER_WINDOW)
    clean_reel_id = str(payload.reel_id or "").strip()
    if not clean_reel_id or len(clean_reel_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid reel_id.")

    def write_feedback(conn: Any) -> None:
        learner_id = _resolve_learner_identity(conn, request)
        # Existence check + write under a single transaction so a concurrent
        # delete of the reel can't race us into writing an orphan feedback row.
        exists = fetch_one(
            conn,
            "SELECT id, material_id FROM reels WHERE id = ? LIMIT 1",
            (clean_reel_id,),
        )
        if not exists:
            raise HTTPException(status_code=404, detail="reel_id not found")

        _lock_learner_adaptation(
            conn,
            material_id=str(exists["material_id"]),
            learner_id=learner_id,
        )
        reel_service.record_feedback(
            conn,
            reel_id=clean_reel_id,
            helpful=payload.helpful,
            confusing=payload.confusing,
            rating=payload.rating,
            saved=payload.saved,
            learner_id=learner_id,
        )
        _cancel_stale_active_adaptation_jobs(
            conn,
            material_id=str(exists["material_id"]),
            learner_id=learner_id,
            adaptation_fingerprint=_learner_adaptation_fingerprint(
                conn,
                material_id=str(exists["material_id"]),
                learner_id=learner_id,
            ),
        )

    _run_adaptive_mutation_transaction(write_feedback)
    return {"status": "ok", "reel_id": clean_reel_id}


@app.post("/api/reels/{reel_id}/progress", response_model=ReelProgressResponse)
async def record_reel_progress(request: Request, reel_id: str, payload: ReelProgressRequest):
    _enforce_rate_limit(
        request, "reel-progress", limit=ASSESSMENT_PROGRESS_RATE_LIMIT_PER_WINDOW
    )
    clean_reel_id = str(reel_id or "").strip()
    if not clean_reel_id or len(clean_reel_id) > 160:
        raise HTTPException(status_code=400, detail="Invalid reel_id.")
    try:
        def work(should_cancel: Callable[[], bool]):
            def write_progress(conn: Any):
                learner_id = _resolve_learner_identity(conn, request)
                return assessment_service.record_progress(
                    conn,
                    learner_id=learner_id,
                    reel_id=clean_reel_id,
                    max_fraction=payload.max_fraction,
                    should_cancel=should_cancel,
                )

            return _run_adaptive_mutation_transaction(
                write_progress,
                should_cancel=should_cancel,
            )

        return await _run_disconnect_cancellable(request, work)
    except AssessmentCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/reels/{reel_id}/scroll", response_model=ReelScrollResponse)
async def record_reel_scroll(request: Request, reel_id: str):
    _enforce_rate_limit(
        request, "reel-scroll", limit=ASSESSMENT_PROGRESS_RATE_LIMIT_PER_WINDOW
    )
    clean_reel_id = str(reel_id or "").strip()
    if not clean_reel_id or len(clean_reel_id) > 160:
        raise HTTPException(status_code=400, detail="Invalid reel_id.")
    try:
        def work(should_cancel: Callable[[], bool]):
            def write_scroll(conn: Any):
                learner_id = _resolve_learner_identity(conn, request)
                return assessment_service.record_scroll(
                    conn,
                    learner_id=learner_id,
                    reel_id=clean_reel_id,
                    should_cancel=should_cancel,
                )

            return _run_adaptive_mutation_transaction(
                write_scroll,
                should_cancel=should_cancel,
            )

        return await _run_disconnect_cancellable(request, work)
    except AssessmentCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/assessments/pending", response_model=AssessmentWrapperResponse)
def pending_assessment(request: Request, material_id: str):
    _enforce_rate_limit(
        request, "assessment-read", limit=ASSESSMENT_ACTION_RATE_LIMIT_PER_WINDOW
    )
    clean_material_id = str(material_id or "").strip()
    if not clean_material_id or len(clean_material_id) > 240:
        raise HTTPException(status_code=400, detail="Invalid material_id.")
    with get_conn() as conn:
        if not fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (clean_material_id,)):
            raise HTTPException(status_code=404, detail="material_id not found")
        learner_id = _resolve_learner_identity(conn, request)
        return assessment_service.pending(
            conn, learner_id=learner_id, material_id=clean_material_id
        )


@app.post("/api/assessments/next", response_model=AssessmentWrapperResponse)
async def next_assessment(request: Request, payload: AssessmentNextRequest):
    _enforce_rate_limit(
        request, "assessment-next", limit=ASSESSMENT_ACTION_RATE_LIMIT_PER_WINDOW
    )
    try:
        def work(should_cancel: Callable[[], bool]):
            def write_session(conn: Any):
                if not fetch_one(
                    conn,
                    "SELECT id FROM materials WHERE id = ?",
                    (payload.material_id,),
                ):
                    raise HTTPException(status_code=404, detail="material_id not found")
                learner_id = _resolve_learner_identity(conn, request)
                return assessment_service.next_session(
                    conn,
                    learner_id=learner_id,
                    material_id=payload.material_id,
                    should_cancel=should_cancel,
                )

            return _run_adaptive_mutation_transaction(
                write_session,
                should_cancel=should_cancel,
            )

        return await _run_disconnect_cancellable(request, work)
    except AssessmentCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc


@app.post(
    "/api/assessments/{session_id}/answer",
    response_model=AssessmentAnswerResponse,
)
def answer_assessment(
    request: Request, session_id: str, payload: AssessmentAnswerRequest
):
    _enforce_rate_limit(
        request, "assessment-answer", limit=ASSESSMENT_ACTION_RATE_LIMIT_PER_WINDOW
    )
    try:
        def write_answer(conn: Any) -> dict[str, Any]:
            learner_id = _resolve_learner_identity(conn, request)
            clean_session_id = str(session_id or "").strip()
            session_scope = fetch_one(
                conn,
                "SELECT material_id FROM assessment_sessions "
                "WHERE id = ? AND learner_id = ?",
                (clean_session_id, learner_id),
            )
            if session_scope:
                _lock_learner_adaptation(
                    conn,
                    material_id=str(session_scope["material_id"]),
                    learner_id=learner_id,
                )
            result = assessment_service.answer(
                conn,
                learner_id=learner_id,
                session_id=clean_session_id,
                question_id=payload.question_id,
                choice_index=payload.choice_index,
            )
            session = result.get("session") or {}
            if session.get("status") == "completed":
                material_id = str(session.get("material_id") or "")
                reel_service.update_level_adjustment(
                    conn, material_id, learner_id
                )
                _cancel_stale_active_adaptation_jobs(
                    conn,
                    material_id=material_id,
                    learner_id=learner_id,
                    adaptation_fingerprint=_learner_adaptation_fingerprint(
                        conn,
                        material_id=material_id,
                        learner_id=learner_id,
                    ),
                )
            return result

        return _run_adaptive_mutation_transaction(write_answer)
    except ValueError as exc:
        detail = str(exc)
        status = 409 if "not pending" in detail or "answered in order" in detail else 404
        raise HTTPException(status_code=status, detail=detail) from exc


@app.post(
    "/api/assessments/{session_id}/snooze",
    response_model=AssessmentSnoozeResponse,
)
def snooze_assessment(request: Request, session_id: str):
    _enforce_rate_limit(
        request, "assessment-snooze", limit=ASSESSMENT_ACTION_RATE_LIMIT_PER_WINDOW
    )
    try:
        with get_conn(transactional=True) as conn:
            learner_id = _resolve_learner_identity(conn, request)
            return assessment_service.snooze(
                conn,
                learner_id=learner_id,
                session_id=str(session_id or "").strip(),
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/community/reels/duration")
def get_community_reel_duration(request: Request, source_url: str):
    _enforce_rate_limit(request, "community-duration", limit=COMMUNITY_DURATION_RATE_LIMIT_PER_WINDOW)
    normalized_url = source_url.strip()
    if not normalized_url:
        raise HTTPException(status_code=400, detail="source_url is required.")
    normalized_url = _normalize_community_duration_source_url(normalized_url)

    cache_hit, cached_duration_sec = _cached_community_reel_duration_sec(normalized_url)
    if cache_hit:
        return {"duration_sec": cached_duration_sec}
    with get_conn(transactional=True) as conn:
        _require_verified_provider_account(conn, request)
    duration_sec = _resolve_community_reel_duration_sec(normalized_url)
    return {"duration_sec": duration_sec}


@app.get("/api/community/sets", response_model=CommunitySetsResponse)
def list_community_sets(request: Request, limit: int = 160):
    safe_limit = max(1, min(limit, 300))
    with get_conn(transactional=True) as conn:
        # Optional auth: anonymous callers still get the list, but they
        # won't see `viewer_liked` / `viewer_disliked` toggles populated.
        viewer_account = _try_get_community_account(conn, request)
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
                dislikes,
                learners,
                updated_label,
                updated_at,
                thumbnail_url,
                featured,
                created_at
            FROM community_sets
            WHERE visibility = ? OR (featured = 1 AND (visibility IS NULL OR visibility = ? OR visibility = ''))
            ORDER BY featured DESC, COALESCE(NULLIF(updated_at, ''), created_at) DESC
            LIMIT ?
            """,
            (PUBLIC_COMMUNITY_VISIBILITY, PUBLIC_COMMUNITY_VISIBILITY, safe_limit),
        )
        viewer_votes = _fetch_viewer_votes_by_set_id(
            conn,
            account_id=str(viewer_account["id"]) if viewer_account else None,
            set_ids=(row.get("id") for row in rows),
        )
    sets = [
        _serialize_community_set(row, viewer_vote=viewer_votes.get(str(row.get("id") or "")))
        for row in rows
    ]
    return {"sets": sets}


@app.get("/api/community/sets/mine", response_model=CommunitySetsResponse)
def list_my_community_sets(request: Request, limit: int = 160):
    safe_limit = max(1, min(limit, 300))
    with get_conn(transactional=True) as conn:
        account = _require_verified_community_account(conn, request)
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
                dislikes,
                learners,
                updated_label,
                updated_at,
                thumbnail_url,
                featured,
                created_at
            FROM community_sets
            WHERE owner_account_id = ?
            ORDER BY COALESCE(NULLIF(updated_at, ''), created_at) DESC
            LIMIT ?
            """,
            (str(account["id"]), safe_limit),
        )
        # The owner is also the viewer for this list — show their own
        # votes on their own sets, so the detail view's thumbs buttons
        # reflect state consistently across both tabs.
        viewer_votes = _fetch_viewer_votes_by_set_id(
            conn,
            account_id=str(account["id"]),
            set_ids=(row.get("id") for row in rows),
        )
    sets = [
        _serialize_community_set(row, viewer_vote=viewer_votes.get(str(row.get("id") or "")))
        for row in rows
    ]
    return {"sets": sets}


@app.get("/api/community/history", response_model=CommunityHistoryResponse)
def get_community_history(request: Request, limit: int = MAX_COMMUNITY_HISTORY_ITEMS):
    safe_limit = max(1, min(limit, MAX_COMMUNITY_HISTORY_ITEMS))
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        rows = fetch_all(
            conn,
            """
            SELECT
                material_id,
                title,
                updated_at,
                starred,
                generation_mode,
                source,
                feed_query,
                active_index,
                active_reel_id,
                recall_json
            FROM community_material_history
            WHERE account_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (str(account["id"]), safe_limit),
        )
        learner_id = f"account:{account['id']}"
        items = [
            _serialize_community_history_item(
                row,
                assessment_stats=assessment_service.history_stats(
                    conn,
                    learner_id=learner_id,
                    material_id=str(row.get("material_id") or ""),
                ),
            )
            for row in rows
        ]
    return {"items": items}


@app.put("/api/community/history", response_model=CommunityHistoryResponse)
def replace_community_history(request: Request, payload: CommunityHistoryReplaceRequest):
    _enforce_rate_limit(request, "community-history", limit=COMMUNITY_HISTORY_RATE_LIMIT_PER_WINDOW)
    normalized_items = _normalize_community_history_items(payload.items)
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        account_id = str(account["id"])
        execute_modify(conn, "DELETE FROM community_material_history WHERE account_id = ?", (account_id,))
        for item in normalized_items:
            insert(
                conn,
                "community_material_history",
                {
                    "account_id": account_id,
                    "material_id": item["material_id"],
                    "title": item["title"],
                    "updated_at": item["updated_at"],
                    "starred": item["starred"],
                    "generation_mode": item["generation_mode"],
                    "source": item["source"],
                    "feed_query": item["feed_query"],
                    "active_index": item["active_index"],
                    "active_reel_id": item["active_reel_id"],
                    "recall_json": item["recall_json"],
                },
            )
        learner_id = f"account:{account_id}"
        items = [
            _serialize_community_history_item(
                row,
                assessment_stats=assessment_service.history_stats(
                    conn,
                    learner_id=learner_id,
                    material_id=str(row.get("material_id") or ""),
                ),
            )
            for row in normalized_items
        ]
    return {"items": items}


@app.get("/api/community/settings", response_model=CommunitySettingsResponse)
def get_community_settings(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        row = fetch_one(
            conn,
            """
            SELECT
                generation_mode,
                default_input_mode,
                min_relevance_threshold,
                start_muted,
                creative_commons_only,
                preferred_video_duration,
                target_clip_duration_sec,
                target_clip_duration_min_sec,
                target_clip_duration_max_sec,
                autoplay_next_reel
            FROM community_account_settings
            WHERE account_id = ?
            """,
            (str(account["id"]),),
        )
    return _serialize_community_settings(row)


@app.put("/api/community/settings", response_model=CommunitySettingsResponse)
def replace_community_settings(request: Request, payload: CommunitySettingsPayload):
    _enforce_rate_limit(request, "community-settings", limit=COMMUNITY_SETTINGS_RATE_LIMIT_PER_WINDOW)
    normalized = _serialize_community_settings(payload.model_dump())
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        upsert(
            conn,
            "community_account_settings",
            {
                "account_id": str(account["id"]),
                "generation_mode": normalized.generation_mode,
                "default_input_mode": normalized.default_input_mode,
                "min_relevance_threshold": normalized.min_relevance_threshold,
                "start_muted": 1 if normalized.start_muted else 0,
                "creative_commons_only": 1 if normalized.creative_commons_only else 0,
                "preferred_video_duration": normalized.preferred_video_duration,
                "target_clip_duration_sec": normalized.target_clip_duration_sec,
                "target_clip_duration_min_sec": normalized.target_clip_duration_min_sec,
                "target_clip_duration_max_sec": normalized.target_clip_duration_max_sec,
                "autoplay_next_reel": 1 if normalized.autoplay_next_reel else 0,
                "updated_at": now_iso(),
            },
            pk="account_id",
        )
    return normalized


@app.post("/api/community/sets", response_model=CommunitySetOut, status_code=201)
def create_community_set(request: Request, payload: CommunitySetCreateRequest):
    _enforce_rate_limit(request, "community-write", limit=COMMUNITY_WRITE_RATE_LIMIT_PER_WINDOW)
    title = payload.title.strip()
    description = payload.description.strip()
    thumbnail_url = payload.thumbnail_url.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Set name is required.")
    if len(title) > 200:
        raise HTTPException(status_code=400, detail="Set name must be 200 characters or fewer.")
    if len(description) < 18:
        raise HTTPException(status_code=400, detail="Description must be at least 18 characters.")
    if len(description) > 5000:
        raise HTTPException(status_code=400, detail="Description must be 5000 characters or fewer.")
    if not thumbnail_url:
        raise HTTPException(status_code=400, detail="Thumbnail is required.")
    thumbnail_url = _normalize_community_thumbnail_url(thumbnail_url)
    if not payload.reels:
        raise HTTPException(status_code=400, detail="Add at least one reel URL.")

    reels_payload: list[dict[str, object]] = []
    for reel in payload.reels:
        source_url = reel.source_url.strip()
        embed_url = reel.embed_url.strip()
        if not source_url or not embed_url:
            continue
        source_url, embed_url = _validate_community_reel_urls(
            platform=reel.platform,
            source_url=source_url,
            embed_url=embed_url,
        )
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
    set_id = f"user-set-{uuid.uuid4()}"
    created_at = now_iso()
    updated_label = "Last Edited: just now"

    with get_conn(transactional=True) as conn:
        account = _require_verified_community_account(conn, request)
        curator = _community_set_curator_for_account(account)
        owner_key_hash = _community_owner_hash_from_request_optional(request)
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
                "dislikes": 0,
                "learners": 1,
                "updated_label": updated_label,
                "thumbnail_url": thumbnail_url,
                "owner_key_hash": owner_key_hash,
                "owner_account_id": str(account["id"]),
                "visibility": DEFAULT_COMMUNITY_VISIBILITY,
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
                dislikes,
                learners,
                updated_label,
                updated_at,
                created_at,
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
def update_community_set(request: Request, set_id: str, payload: CommunitySetUpdateRequest):
    _enforce_rate_limit(request, "community-write", limit=COMMUNITY_WRITE_RATE_LIMIT_PER_WINDOW)
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
    if len(title) > 200:
        raise HTTPException(status_code=400, detail="Set name must be 200 characters or fewer.")
    if len(description) < 18:
        raise HTTPException(status_code=400, detail="Description must be at least 18 characters.")
    if len(description) > 5000:
        raise HTTPException(status_code=400, detail="Description must be 5000 characters or fewer.")
    if not thumbnail_url:
        raise HTTPException(status_code=400, detail="Thumbnail is required.")
    thumbnail_url = _normalize_community_thumbnail_url(thumbnail_url)
    if not payload.reels:
        raise HTTPException(status_code=400, detail="Add at least one reel URL.")

    reels_payload: list[dict[str, object]] = []
    seen_reel_ids: set[str] = set()
    for reel in payload.reels:
        source_url = reel.source_url.strip()
        embed_url = reel.embed_url.strip()
        if not source_url or not embed_url:
            continue
        source_url, embed_url = _validate_community_reel_urls(
            platform=reel.platform,
            source_url=source_url,
            embed_url=embed_url,
        )
        t_start_sec = _normalize_clip_seconds(reel.t_start_sec)
        t_end_sec = _normalize_clip_seconds(reel.t_end_sec)
        if t_end_sec is not None:
            start_for_validation = t_start_sec if t_start_sec is not None else 0.0
            if t_end_sec <= start_for_validation:
                raise HTTPException(status_code=400, detail="Clip end time must be greater than start time.")
        existing_reel_id = _normalize_optional_community_reel_id(reel.id)
        reel_id = existing_reel_id if existing_reel_id and existing_reel_id not in seen_reel_ids else str(uuid.uuid4())
        seen_reel_ids.add(reel_id)
        reels_payload.append(
            {
                "id": reel_id,
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
    updated_label = "Last Edited: just now"

    with get_conn(transactional=True) as conn:
        account = _require_verified_community_account(conn, request)
        existing_rows = fetch_all(
            conn,
            """
            SELECT
                id,
                curator,
                owner_key_hash,
                owner_account_id,
                visibility,
                likes,
                dislikes,
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
        stored_owner_account_id = _require_community_set_owner_access(
            existing.get("owner_account_id"),
            str(account["id"]),
            action="edit",
        )

        curator = _community_set_curator_for_account(account)
        created_at = str(existing.get("created_at") or "").strip() or updated_at
        stored_owner_key_hash = str(existing.get("owner_key_hash") or "").strip() or None
        visibility = str(existing.get("visibility") or "").strip() or DEFAULT_COMMUNITY_VISIBILITY

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
                "dislikes": max(0, _to_int(existing.get("dislikes"), 0)),
                "learners": max(0, _to_int(existing.get("learners"), 1)),
                "updated_label": updated_label,
                "thumbnail_url": thumbnail_url,
                "owner_key_hash": stored_owner_key_hash,
                "owner_account_id": stored_owner_account_id,
                "visibility": visibility,
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
                dislikes,
                learners,
                updated_label,
                updated_at,
                created_at,
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


@app.post("/api/community/sets/bulk-delete", status_code=204)
def delete_community_sets(request: Request, payload: CommunitySetsDeleteRequest):
    _enforce_rate_limit(request, "community-write", limit=COMMUNITY_WRITE_RATE_LIMIT_PER_WINDOW)
    normalized_set_ids = list(dict.fromkeys(set_id.strip() for set_id in payload.set_ids))
    if any(not set_id for set_id in normalized_set_ids):
        raise HTTPException(status_code=400, detail="Every set_id is required.")
    if any(not set_id.startswith("user-set-") for set_id in normalized_set_ids):
        raise HTTPException(status_code=403, detail="Only user-created sets can be deleted.")

    placeholders = ", ".join("?" for _ in normalized_set_ids)
    with get_conn(transactional=True) as conn:
        account = _require_verified_community_account(conn, request)
        account_id = str(account["id"])
        rows = fetch_all(
            conn,
            f"""
            SELECT id, owner_account_id, featured
            FROM community_sets
            WHERE id IN ({placeholders})
            """,
            tuple(normalized_set_ids),
        )
        rows_by_id = {str(row.get("id") or ""): row for row in rows}
        for set_id in normalized_set_ids:
            existing = rows_by_id.get(set_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Community set not found.")
            if _to_int(existing.get("featured"), 0) != 0:
                raise HTTPException(status_code=403, detail="Featured sets cannot be deleted.")
            _require_community_set_owner_access(
                existing.get("owner_account_id"),
                account_id,
                action="delete",
            )

        execute_modify(
            conn,
            f"DELETE FROM community_set_votes WHERE set_id IN ({placeholders})",
            tuple(normalized_set_ids),
        )
        execute_modify(
            conn,
            f"DELETE FROM community_starred_sets WHERE set_id IN ({placeholders})",
            tuple(normalized_set_ids),
        )
        deleted_count = execute_modify(
            conn,
            f"""
            DELETE FROM community_sets
            WHERE owner_account_id = ?
              AND featured = 0
              AND id IN ({placeholders})
            """,
            (account_id, *normalized_set_ids),
        )
        if deleted_count != len(normalized_set_ids):
            raise HTTPException(
                status_code=409,
                detail="One or more community sets changed before they could be deleted.",
            )

    return Response(status_code=204)


def _delete_community_set_impl(request: Request, set_id: str) -> Response:
    normalized_set_id = set_id.strip()
    if not normalized_set_id:
        raise HTTPException(status_code=400, detail="set_id is required.")
    if not normalized_set_id.startswith("user-set-"):
        raise HTTPException(status_code=403, detail="Only user-created sets can be deleted.")
    with get_conn(transactional=True) as conn:
        account = _require_verified_community_account(conn, request)
        existing_rows = fetch_all(
            conn,
            """
            SELECT
                id,
                owner_account_id,
                featured
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
            raise HTTPException(status_code=403, detail="Featured sets cannot be deleted.")
        _require_community_set_owner_access(
            existing.get("owner_account_id"),
            str(account["id"]),
            action="delete",
        )

        deleted_count = execute_modify(conn, "DELETE FROM community_sets WHERE id = ?", (normalized_set_id,))

        if deleted_count <= 0:
            still_exists = fetch_all(
                conn,
                """
                SELECT id
                FROM community_sets
                WHERE id = ?
                LIMIT 1
                """,
                (normalized_set_id,),
            )
            if still_exists:
                raise HTTPException(status_code=500, detail="Community set delete did not persist.")

    return Response(status_code=204)


@app.delete("/api/community/sets/{set_id}", status_code=204)
def delete_community_set(request: Request, set_id: str):
    _enforce_rate_limit(request, "community-write", limit=COMMUNITY_WRITE_RATE_LIMIT_PER_WINDOW)
    return _delete_community_set_impl(request, set_id)


@app.post("/api/community/sets/{set_id}/delete", status_code=204)
def delete_community_set_via_post(request: Request, set_id: str):
    # Fallback for deployments/proxies that block DELETE with 405.
    _enforce_rate_limit(request, "community-write", limit=COMMUNITY_WRITE_RATE_LIMIT_PER_WINDOW)
    return _delete_community_set_impl(request, set_id)


@app.post(
    "/api/community/sets/{set_id}/feedback",
    response_model=CommunitySetFeedbackResponse,
)
def record_community_set_feedback(
    request: Request,
    set_id: str,
    payload: CommunitySetFeedbackRequest,
):
    """Upsert the caller's like/dislike vote on a community set and
    return the freshly-recomputed aggregate totals along with the
    viewer's new vote state.

    - ``liked=True, disliked=False`` → store a 'like' row
    - ``liked=False, disliked=True`` → store a 'dislike' row
    - ``liked=False, disliked=False`` → clear any prior vote
    - ``liked=True, disliked=True`` → 400 (mutually exclusive)

    Aggregate ``likes`` / ``dislikes`` columns on ``community_sets``
    are recomputed from the per-user ``community_set_votes`` table on
    every write, so the totals can never drift away from the row-level
    truth.
    """
    _enforce_rate_limit(request, "community-set-feedback", limit=120)

    if payload.liked and payload.disliked:
        raise HTTPException(
            status_code=400,
            detail="A set cannot be liked and disliked at the same time.",
        )

    normalized_set_id = set_id.strip()
    if not normalized_set_id:
        raise HTTPException(status_code=400, detail="set_id is required.")

    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        account_id = str(account["id"])

        existing = fetch_one(
            conn,
            "SELECT id FROM community_sets WHERE id = ? LIMIT 1",
            (normalized_set_id,),
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Community set not found.")

        # Clear any prior vote for (account, set), then write the new
        # one. DELETE-then-INSERT is simpler than a cross-dialect UPSERT
        # here, and stays correct because the whole block is inside a
        # single transaction.
        execute_modify(
            conn,
            "DELETE FROM community_set_votes WHERE account_id = ? AND set_id = ?",
            (account_id, normalized_set_id),
        )

        new_vote: str | None = None
        if payload.liked:
            new_vote = "like"
        elif payload.disliked:
            new_vote = "dislike"

        if new_vote is not None:
            insert(
                conn,
                "community_set_votes",
                {
                    "account_id": account_id,
                    "set_id": normalized_set_id,
                    "vote": new_vote,
                    "created_at": now_iso(),
                },
            )

        # Recount authoritative totals from the votes table.
        totals = fetch_one(
            conn,
            """
            SELECT
                SUM(CASE WHEN vote = 'like' THEN 1 ELSE 0 END) AS like_total,
                SUM(CASE WHEN vote = 'dislike' THEN 1 ELSE 0 END) AS dislike_total
            FROM community_set_votes
            WHERE set_id = ?
            """,
            (normalized_set_id,),
        )
        likes_total = max(0, _to_int((totals or {}).get("like_total"), 0))
        dislikes_total = max(0, _to_int((totals or {}).get("dislike_total"), 0))

        execute_modify(
            conn,
            "UPDATE community_sets SET likes = ?, dislikes = ? WHERE id = ?",
            (likes_total, dislikes_total, normalized_set_id),
        )

    return CommunitySetFeedbackResponse(
        status="ok",
        set_id=normalized_set_id,
        likes=likes_total,
        dislikes=dislikes_total,
        viewer_liked=new_vote == "like",
        viewer_disliked=new_vote == "dislike",
    )


# ---------------------------------------------------------------------------
# Community Starred Sets
# ---------------------------------------------------------------------------


@app.get("/api/community/starred-sets", response_model=CommunityStarredSetsResponse)
def get_community_starred_sets(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        rows = fetch_all(conn, "SELECT set_id FROM community_starred_sets WHERE account_id = ?", (str(account["id"]),))
    return CommunityStarredSetsResponse(set_ids=[r["set_id"] for r in rows])


@app.put("/api/community/starred-sets", response_model=CommunityStarredSetsResponse)
def replace_community_starred_sets(request: Request, payload: CommunityStarredSetsPayload):
    _enforce_rate_limit(request, "community-starred-sets", limit=90)
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        aid = str(account["id"])
        execute_modify(conn, "DELETE FROM community_starred_sets WHERE account_id = ?", (aid,))
        ts = now_iso()
        for sid in payload.set_ids[:200]:  # cap at 200
            insert(conn, "community_starred_sets", {"account_id": aid, "set_id": sid.strip(), "created_at": ts})
    return CommunityStarredSetsResponse(set_ids=payload.set_ids[:200])


# ---------------------------------------------------------------------------
# Community Feed Snapshots
# ---------------------------------------------------------------------------


@app.get("/api/community/feed-snapshots")
def get_community_feed_snapshots(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        rows = fetch_all(conn, "SELECT material_key, snapshot_json FROM community_feed_snapshots WHERE account_id = ? ORDER BY updated_at DESC LIMIT 50", (str(account["id"]),))
    snapshots = {}
    for r in rows:
        try:
            snapshots[r["material_key"]] = json.loads(r["snapshot_json"])
        except Exception:
            pass
    return {"snapshots": snapshots}


@app.put("/api/community/feed-snapshots/{key:path}")
def upsert_community_feed_snapshot(key: str, request: Request, payload: CommunityFeedSnapshotPayload):
    _enforce_rate_limit(request, "community-feed-snapshots", limit=90)
    clean_key = key.strip()[:200]
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        aid = str(account["id"])
        upsert(conn, "community_feed_snapshots", {
            "account_id": aid,
            "material_key": clean_key,
            "snapshot_json": json.dumps(payload.snapshot),
            "updated_at": now_iso(),
        }, pk=["account_id", "material_key"])
    return {"status": "ok"}


@app.delete("/api/community/feed-snapshots/{key:path}")
def delete_community_feed_snapshot(key: str, request: Request):
    clean_key = key.strip()[:200]
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        execute_modify(conn, "DELETE FROM community_feed_snapshots WHERE account_id = ? AND material_key = ?", (str(account["id"]), clean_key))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Community Drafts
# ---------------------------------------------------------------------------


@app.get("/api/community/drafts")
def get_community_drafts(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        rows = fetch_all(conn, "SELECT draft_key, draft_json FROM community_drafts WHERE account_id = ? ORDER BY updated_at DESC LIMIT 50", (str(account["id"]),))
    drafts = {}
    for r in rows:
        try:
            drafts[r["draft_key"]] = json.loads(r["draft_json"])
        except Exception:
            pass
    return {"drafts": drafts}


@app.put("/api/community/drafts/{key:path}")
def upsert_community_draft(key: str, request: Request, payload: CommunityDraftPayload):
    _enforce_rate_limit(request, "community-drafts", limit=90)
    clean_key = key.strip()[:200]
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        upsert(conn, "community_drafts", {
            "account_id": str(account["id"]),
            "draft_key": clean_key,
            "draft_json": json.dumps(payload.draft),
            "updated_at": now_iso(),
        }, pk=["account_id", "draft_key"])
    return {"status": "ok"}


@app.delete("/api/community/drafts/{key:path}")
def delete_community_draft(key: str, request: Request):
    clean_key = key.strip()[:200]
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        execute_modify(conn, "DELETE FROM community_drafts WHERE account_id = ? AND draft_key = ?", (str(account["id"]), clean_key))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Community Material Seeds
# ---------------------------------------------------------------------------

MAX_MATERIAL_SEEDS = 120


@app.get("/api/community/material-seeds")
def get_community_material_seeds(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        rows = fetch_all(conn, "SELECT material_id, seed_json FROM community_material_seeds WHERE account_id = ? ORDER BY updated_at DESC LIMIT ?", (str(account["id"]), MAX_MATERIAL_SEEDS))
    seeds = {}
    for r in rows:
        try:
            seeds[r["material_id"]] = json.loads(r["seed_json"])
        except Exception:
            pass
    return {"seeds": seeds}


@app.put("/api/community/material-seeds")
def replace_community_material_seeds(request: Request, payload: CommunityMaterialSeedsPayload):
    _enforce_rate_limit(request, "community-material-seeds", limit=90)
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        aid = str(account["id"])
        execute_modify(conn, "DELETE FROM community_material_seeds WHERE account_id = ?", (aid,))
        ts = now_iso()
        for mid, seed in list(payload.seeds.items())[:MAX_MATERIAL_SEEDS]:
            insert(conn, "community_material_seeds", {
                "account_id": aid,
                "material_id": mid.strip()[:500],
                "seed_json": json.dumps(seed),
                "updated_at": seed.get("updated_at", ts) if isinstance(seed, dict) else ts,
            })
    return {"seeds": payload.seeds}


# ---------------------------------------------------------------------------
# Community Material Groups
# ---------------------------------------------------------------------------

MAX_MATERIAL_GROUPS = 80


@app.get("/api/community/material-groups")
def get_community_material_groups(request: Request):
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        rows = fetch_all(conn, "SELECT group_id, group_json FROM community_material_groups WHERE account_id = ? ORDER BY updated_at DESC LIMIT ?", (str(account["id"]), MAX_MATERIAL_GROUPS))
    groups = {}
    for r in rows:
        try:
            groups[r["group_id"]] = json.loads(r["group_json"])
        except Exception:
            pass
    return {"groups": groups}


@app.put("/api/community/material-groups")
def replace_community_material_groups(request: Request, payload: CommunityMaterialGroupsPayload):
    _enforce_rate_limit(request, "community-material-groups", limit=90)
    with get_conn(transactional=True) as conn:
        account = _require_authenticated_community_account(conn, request)
        aid = str(account["id"])
        execute_modify(conn, "DELETE FROM community_material_groups WHERE account_id = ?", (aid,))
        ts = now_iso()
        for gid, group in list(payload.groups.items())[:MAX_MATERIAL_GROUPS]:
            insert(conn, "community_material_groups", {
                "account_id": aid,
                "group_id": gid.strip()[:500],
                "group_json": json.dumps(group),
                "updated_at": group.get("updated_at", ts) if isinstance(group, dict) else ts,
            })
    return {"groups": payload.groups}


# ---------------------------------------------------------------------------
# Admin: server-side simulation test
# ---------------------------------------------------------------------------

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
