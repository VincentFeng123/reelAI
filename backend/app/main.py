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
from collections.abc import Iterable
from typing import Any, Callable, Literal
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute

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
    now_iso,
    upsert,
)
from .models import (
    AssessmentAnswerRequest,
    AssessmentAnswerResponse,
    AssessmentNextRequest,
    AssessmentSnoozeResponse,
    AssessmentWrapperResponse,
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
from .services.assessments import AssessmentCancelledError, AssessmentService
from .services.email import send_welcome_email
from .services.embeddings import EmbeddingService
from .services.material_intelligence import MaterialIntelligenceService
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
from .clip_engine.errors import CancellationError as ClipEngineCancellationError, EngineError as ClipEngineError
from .clip_engine.errors import ProviderError as ClipEngineProviderError
from .clip_engine.provider_cache import DatabaseProviderCache
from .clip_engine.provider_cache import normalize_filters as normalize_provider_filters
from .clip_engine.provider_cache import search_cache_key
from .clip_engine.provider_runtime import GenerationContext, ProviderUsageRecord
from .clip_engine.clipper.supadata_client import fetch_transcript_artifact
from .clip_engine.supadata_search import search_one as supadata_search_one
from .clip_engine.metadata import canonicalize_youtube_url
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
    JobLeaseLostError,
    TERMINAL_STATUSES as GENERATION_TERMINAL_STATUSES,
    append_event as append_generation_event,
    build_request_key as build_durable_request_key,
    cancellation_requested as generation_cancellation_requested,
    expire_stale_queued_job as expire_stale_generation_job,
    find_active_job as find_active_generation_job,
    find_completed_job as find_completed_generation_job,
    get_job as get_generation_job,
    heartbeat_job as heartbeat_generation_job,
    lease_next_job,
    material_content_fingerprint,
    record_provider_usage,
    replay_events as replay_generation_events,
    request_cancellation as request_generation_cancellation,
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
    0.1,
    min(30.0, float(settings.generation_job_poll_sec)),
)
GENERATION_WORKER_COUNT = max(
    1,
    min(8, int(settings.generation_job_worker_count)),
)
_generation_worker_ids = tuple(
    f"worker-{uuid.uuid4()}" for _index in range(GENERATION_WORKER_COUNT)
)
_generation_worker_id = _generation_worker_ids[0]
_generation_worker_stop = threading.Event()
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


# Cumulative upper bound on reels stored per material across all feed pages.
# The per-request generation cap in ReelService already bounds each individual
# call, but without this ceiling a client that keeps paginating indefinitely
# can drive total storage to unbounded growth (200 pages * 25 limit = 5,000
# reels per material in the worst case). 300 comfortably covers typical study
# sessions (3-50 reels) while guarding against runaway pagination.
MAX_REELS_PER_MATERIAL = 300

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
REELS_GENERATE_RATE_LIMIT_PER_WINDOW = 36
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
    ranked: list[tuple[int, float, int, dict[str, Any]]] = []
    for original_index, reel in enumerate(reels):
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

        duration_fit = "in_range"
        distance = 0.0
        if target_clip_duration_min_sec is not None and clip_duration_value < float(
            target_clip_duration_min_sec
        ):
            duration_fit = "shorter"
            distance = float(target_clip_duration_min_sec) - clip_duration_value
        elif target_clip_duration_max_sec is not None and clip_duration_value > float(
            target_clip_duration_max_sec
        ):
            duration_fit = "longer"
            distance = clip_duration_value - float(target_clip_duration_max_sec)

        normalized["duration_preference_met"] = duration_fit == "in_range"
        normalized["duration_fit"] = duration_fit
        ranked.append((0 if duration_fit == "in_range" else 1, distance, original_index, normalized))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in ranked]


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
    lowered = f" {text.lower()} "
    text_tokens = set(re.findall(r"[a-z0-9\+#]+", text.lower()))
    for raw_term in terms:
        normalized = " ".join(re.findall(r"[a-z0-9\+#]+", str(raw_term or "").lower()))
        if not normalized:
            continue
        if f" {normalized} " in lowered:
            return True
        tokens = normalized.split()
        if len(tokens) > 1 and set(tokens).issubset(text_tokens):
            return True
    return False


def _request_matched_anchor_terms(text: str, terms: list[str]) -> list[str]:
    lowered = f" {text.lower()} "
    text_tokens = set(re.findall(r"[a-z0-9\+#]+", text.lower()))
    matches: list[str] = []
    seen: set[str] = set()
    for raw_term in terms:
        candidate = str(raw_term or "").strip()
        normalized = " ".join(re.findall(r"[a-z0-9\+#]+", candidate.lower()))
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
    left_tokens = set(re.findall(r"[a-z0-9\+#]+", left_text))
    right_tokens = set(re.findall(r"[a-z0-9\+#]+", right_text))
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
            clip_duration_sec = max(0.0, float(reel.get("clip_duration_sec") or 0.0))
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
                ((0 < video_duration_sec <= 3 * 60) or (0 < clip_duration_sec <= 75.0))
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


def _count_generation_reels(conn, generation_id: str) -> int:
    try:
        rows = fetch_all(
            conn,
            "SELECT search_context_json FROM reels WHERE generation_id = ?",
            (generation_id,),
        )
    except Exception as exc:
        if "search_context_json" not in str(exc).lower():
            raise
        legacy = fetch_one(
            conn,
            "SELECT COUNT(*) AS reel_count FROM reels WHERE generation_id = ?",
            (generation_id,),
        )
        return max(0, int((legacy or {}).get("reel_count") or 0))
    count = 0
    for row in rows:
        try:
            context = json.loads(str(row.get("search_context_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            context = {}
        if isinstance(context, dict):
            surface_eligible = context.get("surface_eligible", True)
            if isinstance(surface_eligible, str):
                surface_eligible = surface_eligible.strip().lower() in {
                    "1", "true", "yes", "on",
                }
            if not surface_eligible:
                continue
            if str(context.get("boundary_status") or "").strip().lower() == "unavailable":
                continue
        count += 1
    return count


def _fetch_generation_row(conn, generation_id: str | None) -> dict[str, Any] | None:
    if not generation_id:
        return None
    return fetch_one(conn, "SELECT * FROM reel_generations WHERE id = ?", (generation_id,))


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
) -> str:
    generation_id = str(uuid.uuid4())
    upsert(
        conn,
        "reel_generations",
        {
            "id": generation_id,
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
    return generation_id


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


def _reel_identity_key(reel: dict[str, Any]) -> tuple[str, str]:
    reel_id = str(reel.get("reel_id") or "").strip()
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


def _response_generation_ids(conn, generation_id: str | None) -> list[str]:
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
        source_generation_id = str(generation_row.get("source_generation_id") or "").strip()
        if source_generation_id:
            collect(source_generation_id)
        ordered.append(str(generation_row.get("id") or current_generation_id))

    collect(generation_id)
    return ordered


def _verified_cross_request_source_generation(
    conn,
    *,
    material_id: str,
    learner_id: str,
    request_key: str,
    concept_id: str | None,
) -> str | None:
    """Return the newest other-request inventory only when its whole chain is verified."""
    candidate = fetch_one(
        conn,
        """
        SELECT result_generation_id
        FROM reel_generation_jobs
        WHERE material_id = ?
          AND learner_id = ?
          AND request_key <> ?
          AND ((? IS NULL AND concept_id IS NULL) OR concept_id = ?)
          AND status IN ('completed', 'partial')
          AND result_generation_id IS NOT NULL
          AND TRIM(result_generation_id) <> ''
        ORDER BY completed_at DESC, updated_at DESC, created_at DESC, id DESC
        LIMIT 1
        """,
        (material_id, learner_id, request_key, concept_id, concept_id),
    )
    generation_id = str((candidate or {}).get("result_generation_id") or "").strip()
    if not generation_id:
        return None
    generation_ids = _response_generation_ids(conn, generation_id)
    if not generation_ids:
        return None

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
        return None

    reel_rows = fetch_all(
        conn,
        f"SELECT search_context_json FROM reels WHERE generation_id IN ({placeholders})",
        tuple(generation_ids),
    )
    verified_surfaceable_count = 0
    for row in reel_rows:
        try:
            context = json.loads(str(row.get("search_context_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(context, dict):
            return None
        if "surface_eligible" not in context:
            return None
        surface_eligible = context.get("surface_eligible")
        if isinstance(surface_eligible, str):
            surface_eligible = surface_eligible.strip().lower() in {
                "1", "true", "yes", "on",
            }
        if not surface_eligible:
            continue
        if str(context.get("boundary_status") or "").strip().lower() != "verified":
            return None
        verified_surfaceable_count += 1
    return generation_id if verified_surfaceable_count else None


def _finalize_request_reel_order(
    conn,
    *,
    material_id: str,
    learner_id: str,
    rows: list[dict[str, Any]],
    previous_video_id: str,
) -> list[dict[str, Any]]:
    """Do not reapply legacy chronology to an already versioned feed order."""
    versioned_order = bool(rows) and all(
        bool(row.get("_selection_ordered")) for row in rows
    )
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("_selection_ordered", None)
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
    generation_ids = _response_generation_ids(conn, generation_id)
    if generation_ids:
        ranked_batches = [
            reel_service.ranked_feed(
                conn,
                material_id,
                fast_mode=fast_mode,
                generation_id=current_generation_id,
                page_hint=page,
                learner_id=learner_id,
                exclusions_fingerprint=exclusions_fingerprint,
                content_fingerprint=content_fingerprint,
            )
            for current_generation_id in generation_ids
            if current_generation_id
        ]
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
        )
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


def _persist_generation_provider_usage(job_id: str, record: ProviderUsageRecord) -> None:
    """Store every provider response instead of treating one rate-limit hit as a generation."""
    with get_conn(transactional=True) as conn:
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
        )


def _job_request_params(job_row: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(str(job_row.get("request_params_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _generation_job_reels(conn, job_row: dict[str, Any]) -> list[dict[str, Any]]:
    generation_id = str(job_row.get("result_generation_id") or "").strip()
    if not generation_id:
        return []
    params = _job_request_params(job_row)
    requested = max(1, min(MAX_REELS_PER_MATERIAL, int(params.get("num_reels") or 20)))
    return _ranked_request_reels(
        conn,
        material_id=str(job_row.get("material_id") or ""),
        fast_mode=str(params.get("generation_mode") or "slow") == "fast",
        generation_id=generation_id,
        min_relevance=_normalize_min_relevance(params.get("min_relevance")),
        preferred_video_duration=_normalize_preferred_video_duration(
            str(params.get("preferred_video_duration") or "any")
        ),
        target_clip_duration_sec=int(params.get("target_clip_duration_sec") or 55),
        target_clip_duration_min_sec=(
            int(params["target_clip_duration_min_sec"])
            if params.get("target_clip_duration_min_sec") is not None
            else None
        ),
        target_clip_duration_max_sec=(
            int(params["target_clip_duration_max_sec"])
            if params.get("target_clip_duration_max_sec") is not None
            else None
        ),
        exclude_video_ids=list(params.get("exclude_video_ids") or []),
        page=1,
        limit=requested,
        learner_id=str(job_row.get("learner_id") or LEGACY_LEARNER_ID),
    )[:requested]


def _generation_job_status_payload(conn, job_row: dict[str, Any]) -> dict[str, Any]:
    try:
        usage = json.loads(str(job_row.get("usage_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        usage = {}
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
    terminal = str(job_row.get("status") or "") in GENERATION_TERMINAL_STATUSES
    return {
        "job_id": str(job_row.get("id") or ""),
        "status": str(job_row.get("status") or "queued"),
        "phase": str(job_row.get("phase") or ""),
        "progress": float(job_row.get("progress") or 0.0),
        "attempt_count": int(job_row.get("attempt_count") or 0),
        "max_attempts": int(job_row.get("max_attempts") or 2),
        "lease_expires_at": _normalize_datetime_for_api(job_row.get("lease_expires_at")),
        "heartbeat_at": _normalize_datetime_for_api(job_row.get("heartbeat_at")),
        "deadline_at": _normalize_datetime_for_api(job_row.get("deadline_at")),
        "material_id": str(job_row.get("material_id") or ""),
        "request_key": str(job_row.get("request_key") or ""),
        "result_generation_id": str(job_row.get("result_generation_id") or "") or None,
        "model_used": str(job_row.get("model_used") or "") or None,
        "quality_degraded": bool(job_row.get("quality_degraded")),
        "usage": usage if isinstance(usage, dict) else {},
        "error": error,
        "reels": _generation_job_reels(conn, job_row) if terminal else [],
        "created_at": _normalize_datetime_for_api(job_row.get("created_at")),
        "started_at": _normalize_datetime_for_api(job_row.get("started_at")),
        "completed_at": _normalize_datetime_for_api(job_row.get("completed_at")),
    }


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


def _run_leased_generation_job(
    job_row: dict[str, Any],
    worker_stop: threading.Event | None = None,
) -> None:
    job_id = str(job_row.get("id") or "")
    lease_owner = str(job_row.get("lease_owner") or "")
    params = _job_request_params(job_row)
    mode: Literal["fast", "slow"] = "fast" if params.get("generation_mode") == "fast" else "slow"
    requested_count = max(
        1,
        min(MAX_REELS_PER_MATERIAL, int(params.get("num_reels") or 20)),
    )
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
    heartbeat_thread.start()

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

    context = GenerationContext(
        mode,
        generation_id=job_id,
        usage_sink=lambda record: _persist_generation_provider_usage(job_id, record),
        cache_store=DatabaseProviderCache(),
    )
    generation_id = ""
    try:
        with get_conn(transactional=True) as setup_conn:
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
            generation_id = str(job_row.get("result_generation_id") or "").strip()
            if not _fetch_generation_row(setup_conn, generation_id):
                source_generation_id = (
                    str(job_row.get("source_generation_id") or "").strip() or None
                )
                generation_id = _create_generation_row(
                    setup_conn,
                    material_id=material_id,
                    concept_id=concept_id,
                    request_key=str(job_row.get("request_key") or ""),
                    generation_mode=mode,
                    retrieval_profile="unified",
                    source_generation_id=source_generation_id,
                )
                attached_at = now_iso()
                attached = execute_modify(
                    setup_conn,
                    "UPDATE reel_generation_jobs SET result_generation_id = ?, updated_at = ? "
                    "WHERE id = ? AND status = 'running' AND lease_owner = ? "
                    "AND cancel_requested = 0 AND lease_expires_at > ? "
                    "AND (deadline_at IS NULL OR deadline_at > ?)",
                    (
                        generation_id,
                        attached_at,
                        job_id,
                        lease_owner,
                        attached_at,
                        attached_at,
                    ),
                )
                if not attached:
                    raise JobLeaseLostError(
                        f"generation job lease is no longer active: {job_id}"
                    )

        with get_conn() as conn:
            source_generation_ids = _response_generation_ids(
                conn,
                str(job_row.get("source_generation_id") or "").strip() or None,
            )
            source_reel_count = sum(
                _count_generation_reels(conn, source_id)
                for source_id in source_generation_ids
            )
            source_generation_row = _fetch_generation_row(
                conn,
                str(job_row.get("source_generation_id") or "").strip() or None,
            )
            if (
                source_generation_row
                and str(source_generation_row.get("request_key") or "")
                != str(job_row.get("request_key") or "")
            ):
                source_reel_count = min(
                    source_reel_count,
                    max(0, requested_count - 2),
                )
            emitted: set[tuple[str, str]] = set()
            candidate_event_cap = 2 if mode == "fast" else requested_count

            def on_candidate(reel: dict[str, Any]) -> None:
                if should_cancel():
                    raise GenerationCancelledError("Generation cancelled.")
                identity = _reel_identity_key(reel)
                if identity in emitted or len(emitted) >= candidate_event_cap:
                    return
                emitted.add(identity)
                append_generation_event(
                    conn,
                    job_id=job_id,
                    event_type="candidate",
                    payload={"reel": reel, "provisional": True},
                    lease_owner=lease_owner,
                )

            base_exclusions = list(params.get("exclude_video_ids") or [])

            def run_retrieval_stage(
                *,
                retrieval_profile: Literal["bootstrap", "deep"],
                video_budget: int,
                new_reel_cap: int,
                excluded_video_ids: list[str],
                analyzed_video_ids: set[str],
                retrieved_video_ids: set[str],
            ) -> None:
                reel_service.generate_reels(
                    conn,
                    material_id=material_id,
                    concept_id=concept_id,
                    num_reels=requested_count,
                    creative_commons_only=bool(params.get("creative_commons_only")),
                    exclude_video_ids=excluded_video_ids,
                    exclude_generation_ids=source_generation_ids,
                    fast_mode=mode == "fast",
                    preferred_video_duration=_normalize_preferred_video_duration(
                        str(params.get("preferred_video_duration") or "any")
                    ),
                    target_clip_duration_sec=int(params.get("target_clip_duration_sec") or 55),
                    target_clip_duration_min_sec=params.get("target_clip_duration_min_sec"),
                    target_clip_duration_max_sec=params.get("target_clip_duration_max_sec"),
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
                    retrieved_video_ids=retrieved_video_ids,
                )

            current_count = source_reel_count + _count_generation_reels(
                conn, generation_id
            )
            if current_count < requested_count:
                context.budget.reserve_pass()
                if should_cancel():
                    raise GenerationCancelledError("Generation cancelled.")
                bootstrap_analyzed_video_ids: set[str] = set()
                bootstrap_retrieved_video_ids: set[str] = set()
                try:
                    run_retrieval_stage(
                        retrieval_profile="bootstrap",
                        video_budget=3,
                        new_reel_cap=min(2, requested_count - current_count),
                        excluded_video_ids=base_exclusions,
                        analyzed_video_ids=bootstrap_analyzed_video_ids,
                        retrieved_video_ids=bootstrap_retrieved_video_ids,
                    )
                except ClipEngineProviderError as exc:
                    deadline_exhausted = (
                        exc.code == "provider_transient"
                        and exc.operation in {"search", "transcript"}
                        and str(exc.detail or "").strip().casefold()
                        == "generation deadline exceeded"
                    )
                    if not deadline_exhausted:
                        raise
                    # The provisional stage has a strict wall-clock ceiling.
                    # Slow jobs may still recover through deep retrieval; Fast
                    # jobs truthfully finish partial/exhausted.
                    logger.info(
                        "bootstrap deadline exhausted job_id=%s operation=%s",
                        job_id,
                        exc.operation,
                    )
                current_count = source_reel_count + _count_generation_reels(
                    conn, generation_id
                )
                if not update_generation_progress(
                    conn,
                    job_id=job_id,
                    lease_owner=lease_owner,
                    phase="ranking" if mode == "fast" else "retrieval",
                    progress=0.85 if mode == "fast" else 0.45,
                    usage=context.usage_payload(),
                ):
                    raise JobLeaseLostError(
                        f"generation job lease is no longer active: {job_id}"
                    )

                deep_video_budget = min(
                    max(0, 5 - len(bootstrap_analyzed_video_ids)),
                    context.budget.remaining("transcript"),
                    context.budget.remaining("segmentation"),
                )
                if (
                    mode == "slow"
                    and current_count < requested_count
                    and deep_video_budget > 0
                    and context.budget.remaining("search") > 0
                ):
                    deep_exclusions = _normalize_excluded_video_ids(
                        [*base_exclusions, *sorted(bootstrap_retrieved_video_ids)]
                    )
                    run_retrieval_stage(
                        retrieval_profile="deep",
                        video_budget=deep_video_budget,
                        new_reel_cap=requested_count - current_count,
                        excluded_video_ids=deep_exclusions,
                        analyzed_video_ids=set(),
                        retrieved_video_ids=set(),
                    )
                    current_count = source_reel_count + _count_generation_reels(
                        conn, generation_id
                    )
                if mode == "slow":
                    if not update_generation_progress(
                        conn,
                        job_id=job_id,
                        lease_owner=lease_owner,
                        phase="ranking",
                        progress=0.85,
                        usage=context.usage_payload(),
                    ):
                        raise JobLeaseLostError(
                            f"generation job lease is no longer active: {job_id}"
                        )

            cumulative_count = source_reel_count + _count_generation_reels(
                conn, generation_id
            )
            if cumulative_count:
                _activate_generation(
                    conn,
                    material_id=material_id,
                    request_key=str(job_row.get("request_key") or ""),
                    generation_id=generation_id,
                    retrieval_profile="unified",
                )
                refreshed_job = get_generation_job(conn, job_id) or {**job_row, "result_generation_id": generation_id}
                final_reels = _generation_job_reels(conn, refreshed_job)
            else:
                _complete_generation(
                    conn,
                    generation_id=generation_id,
                    retrieval_profile="unified",
                    status="failed",
                    error_text="inventory_exhausted",
                )
                final_reels = []

            usage_records = context.usage()
            stage_counters = context.counters()
            usage_payload = context.usage_payload()
            model_records = [
                row
                for row in usage_records
                if row.get("operation") == "segmentation"
                and bool((row.get("metadata") or {}).get("provider_call"))
            ]
            model_used = str((model_records[-1] if model_records else {}).get("model_used") or "") or None
            quality_degraded = any(bool(row.get("quality_degraded")) for row in model_records)
            append_generation_event(
                conn,
                job_id=job_id,
                event_type="final",
                payload={
                    "reels": final_reels,
                    "generation_id": generation_id if final_reels else None,
                    "authoritative": True,
                },
                lease_owner=lease_owner,
            )
            terminal_status = (
                "completed"
                if len(final_reels) >= requested_count
                else "partial"
                if final_reels
                else "exhausted"
            )
            transition_generation_terminal(
                conn,
                job_id=job_id,
                status=terminal_status,
                result_generation_id=generation_id if final_reels else None,
                lease_owner=lease_owner,
                model_used=model_used,
                quality_degraded=quality_degraded,
                usage=usage_payload,
                error_code="inventory_exhausted" if terminal_status == "exhausted" else None,
                error_message=(
                    _generation_exhaustion_message(stage_counters)
                    if terminal_status == "exhausted"
                    else None
                ),
                error_detail=(
                    {"counters": stage_counters}
                    if terminal_status == "exhausted"
                    else None
                ),
            )
    except GenerationCancelledError:
        with get_conn(transactional=True) as conn:
            if generation_cancellation_requested(conn, job_id):
                request_generation_cancellation(conn, job_id=job_id)
            else:
                # Shutdown, deadline, or a lost lease is not a user
                # cancellation. Leave the durable row for expiry/recovery.
                logger.info("generation worker yielded lease job_id=%s", job_id)
    except ClipEngineProviderError as exc:
        logger.warning("generation provider failure job_id=%s error=%s", job_id, exc.as_dict())
        with get_conn(transactional=True) as conn:
            transition_generation_terminal(
                conn,
                job_id=job_id,
                status="failed",
                result_generation_id=generation_id or None,
                lease_owner=lease_owner,
                usage=context.usage_payload(),
                error_code=exc.code,
                error_message=str(exc),
                error_detail={**exc.as_dict(), "counters": context.counters()},
            )
    except JobLeaseLostError:
        logger.info("generation worker lost lease job_id=%s", job_id)
    except Exception as exc:
        logger.exception("generation job failed job_id=%s", job_id)
        with get_conn(transactional=True) as conn:
            transition_generation_terminal(
                conn,
                job_id=job_id,
                status="failed",
                result_generation_id=generation_id or None,
                lease_owner=lease_owner,
                usage=context.usage_payload(),
                error_code="generation_failed",
                error_message=str(exc),
                error_detail={"counters": context.counters()},
            )
    finally:
        local_stop.set()
        heartbeat_thread.join(timeout=1.0)


def _generation_worker_loop(
    lease_owner: str,
    stop_event: threading.Event,
) -> None:
    worker_id = lease_owner or _generation_worker_id
    while not stop_event.is_set():
        try:
            with get_conn(transactional=True) as conn:
                job_row = lease_next_job(
                    conn,
                    lease_owner=worker_id,
                    lease_seconds=GENERATION_LEASE_SEC,
                )
            if job_row:
                _run_leased_generation_job(job_row, stop_event)
                continue
        except Exception:
            logger.exception("generation worker poll failed")
        stop_event.wait(GENERATION_WORKER_POLL_SEC)


def _start_generation_worker() -> None:
    global _generation_worker_id, _generation_worker_ids, _generation_worker_stop
    global _generation_worker_thread, _generation_worker_threads
    with _generation_worker_lock:
        if any(thread.is_alive() for thread in _generation_worker_threads):
            return
        _generation_worker_stop = threading.Event()
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
        "supadata_configured": bool(os.getenv("SUPADATA_API_KEY", "").strip()),
        "gemini_primary_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "gemini_chat_configured": bool(os.getenv("GEMINI_API_KEY_2", "").strip()),
        "gemini_fallback_model": (
            os.getenv("SEGMENT_PRO_MODEL", "").strip()
            or os.getenv("SEGMENT_FALLBACK_MODEL", "").strip()
            or "gemini-3.1-pro-preview"
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
        execute_modify(conn, "DELETE FROM community_sets WHERE owner_account_id = ?", (account_id,))
        execute_modify(conn, "DELETE FROM community_material_history WHERE account_id = ?", (account_id,))
        execute_modify(conn, "DELETE FROM community_account_settings WHERE account_id = ?", (account_id,))
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
        learner_id = _resolve_learner_identity(conn, request, required=False)
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
                "knowledge_level": normalized_level,
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
        if learner_id:
            reel_service.learner_progress(conn, material_id, learner_id)

    return {
        "material_id": material_id,
        "extracted_concepts": concepts,
    }


@app.patch("/api/materials/{material_id}/level")
def update_material_level(material_id: str, request: Request, payload: MaterialLevelUpdateRequest):
    from .services.knowledge_level import effective_level_target
    with get_conn(transactional=True) as conn:
        row = fetch_one(conn, "SELECT id FROM materials WHERE id = ? LIMIT 1", (material_id,))
        if not row:
            raise HTTPException(status_code=404, detail="material not found")
        learner_id = _resolve_learner_identity(conn, request, required=False)
        if learner_id:
            reel_service.set_learner_level(conn, material_id, learner_id, payload.knowledge_level)
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
    _enforce_rate_limit(request, "reels-generate", limit=REELS_GENERATE_RATE_LIMIT_PER_WINDOW)
    _reject_multi_platform_search(payload.multi_platform_search)
    min_relevance = _normalize_min_relevance(payload.min_relevance)
    safe_video_duration_pref = _normalize_preferred_video_duration(payload.preferred_video_duration)
    safe_target_clip_duration_sec, safe_target_clip_min_sec, safe_target_clip_max_sec = _resolve_target_clip_duration_bounds(
        target_clip_duration_sec=payload.target_clip_duration_sec,
        target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
        target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
    )
    requested_num_reels = max(1, int(payload.num_reels))
    with get_conn(transactional=True) as conn:
        material = fetch_one(
            conn,
            "SELECT id FROM materials WHERE id = ?",
            (payload.material_id,),
        )
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")
        learner_id = _resolve_learner_identity(conn, request)
        learner_progress = reel_service.learner_progress(conn, payload.material_id, learner_id)
        learner_knowledge_level = str(learner_progress.get("selected_level") or "beginner")
        content_fingerprint = material_content_fingerprint(
            conn,
            payload.material_id,
            payload.concept_id,
        )
        request_key = build_durable_request_key(
            material_id=payload.material_id,
            concept_id=payload.concept_id,
            content_fingerprint=content_fingerprint,
            knowledge_level=learner_knowledge_level,
            generation_mode=payload.generation_mode,
            creative_commons_only=payload.creative_commons_only,
            source_duration=safe_video_duration_pref,
            target_clip_duration_sec=safe_target_clip_duration_sec,
            target_clip_duration_min_sec=safe_target_clip_min_sec,
            target_clip_duration_max_sec=safe_target_clip_max_sec,
        )
        source_generation_id: str | None = None
        completed_job = find_completed_generation_job(conn, request_key)
        active_job = find_active_generation_job(conn, request_key)
        if completed_job:
            cached_reels = _generation_job_reels(conn, completed_job)
            if len(cached_reels) >= requested_num_reels:
                return {
                    "reels": cached_reels[:requested_num_reels],
                    "generation_id": str(completed_job.get("result_generation_id") or "") or None,
                    "response_profile": "unified",
                }
            source_generation_id = (
                str(completed_job.get("result_generation_id") or "").strip() or None
            )
        elif not active_job:
            source_generation_id = _verified_cross_request_source_generation(
                conn,
                material_id=payload.material_id,
                learner_id=learner_id,
                request_key=request_key,
                concept_id=payload.concept_id,
            )
        request_params = {
            "material_id": payload.material_id,
            "concept_id": payload.concept_id,
            "num_reels": requested_num_reels,
            "exclude_video_ids": _normalize_excluded_video_ids(payload.exclude_video_ids),
            "creative_commons_only": payload.creative_commons_only,
            "generation_mode": payload.generation_mode,
            "min_relevance": min_relevance,
            "preferred_video_duration": safe_video_duration_pref,
            "target_clip_duration_sec": safe_target_clip_duration_sec,
            "target_clip_duration_min_sec": safe_target_clip_min_sec,
            "target_clip_duration_max_sec": safe_target_clip_max_sec,
            "knowledge_level": learner_knowledge_level,
        }
        job_row, _created = submit_generation_job(
            conn,
            material_id=payload.material_id,
            concept_id=payload.concept_id,
            request_key=request_key,
            content_fingerprint=content_fingerprint,
            learner_id=learner_id,
            request_params=request_params,
            source_generation_id=source_generation_id,
        )
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
        expire_stale_generation_job(conn, job_id=clean_job_id)
        job_row = get_generation_job(conn, clean_job_id)
        if not job_row:
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
        expire_stale_generation_job(conn, job_id=clean_job_id)
        if not get_generation_job(conn, clean_job_id):
            raise HTTPException(status_code=404, detail="job_id not found")

    async def ndjson_events():
        cursor = max(0, int(after_seq or 0))
        while True:
            with get_conn(transactional=True) as conn:
                expire_stale_generation_job(conn, job_id=clean_job_id)
                events = replay_generation_events(
                    conn,
                    job_id=clean_job_id,
                    after_seq=cursor,
                )
                job_row = get_generation_job(conn, clean_job_id)
            for event in events:
                cursor = max(cursor, int(event.get("seq") or 0))
                yield f"{json.dumps(event, default=str, separators=(',', ':'))}\n"
            if any(str(event.get("type") or "") == "terminal" for event in events):
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
        job_row = request_generation_cancellation(
            conn,
            job_id=str(job_id or "").strip(),
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
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestTranscriptionError,
        IngestSegmentationError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
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
        logger.exception("ingest_url crashed for %s", payload.source_url)
        raise _ingest_error_to_http(exc) from exc
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
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestTranscriptionError,
        IngestSegmentationError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
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
        logger.exception("ingest_topic_cut crashed for %s", payload.source_url)
        raise _ingest_error_to_http(exc) from exc
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
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
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
        logger.exception("ingest_search crashed for query=%s", payload.query)
        raise _ingest_error_to_http(exc) from exc
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
        raise HTTPException(status_code=499, detail="Ingestion cancelled.") from exc
    except ClipEngineProviderError as exc:
        raise _provider_error_to_http(exc) from exc
    except (
        IngestUnsupportedSourceError,
        IngestDownloadError,
        IngestServerlessUnavailable,
        IngestRateLimitedError,
    ) as exc:
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
        logger.exception("ingest_feed crashed for %s", payload.feed_url)
        raise _ingest_error_to_http(exc) from exc
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
        evidence = _preflight_evidence(query=query, filters=filters, allow_provider=True)
    except ClipEngineProviderError as exc:
        raise _provider_error_to_http(exc) from exc
    evidence["message"] = (
        "YouTube search found matching candidates."
        if evidence["availability"] == "available"
        else "No matching YouTube candidates were found."
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

    provider_result: dict[str, Any] | None = None
    first_missing = next(
        (index for index, result in enumerate(cached_results) if result["availability"] == "unknown"),
        None,
    )
    if first_missing is not None:
        try:
            provider_result = _preflight_evidence(
                query=queries[first_missing][1],
                filters=filters,
                allow_provider=True,
            )
        except ClipEngineProviderError as exc:
            raise _provider_error_to_http(exc) from exc
        cached_results[first_missing] = provider_result
        if provider_result["availability"] == "available":
            provider_result["materials_checked"] = len(queries)
            provider_result["message"] = "YouTube search found matching candidates."
            provider_result.pop("provider_called", None)
            return provider_result

    known = [result for result in cached_results if result["availability"] != "unknown"]
    all_known = len(known) == len(cached_results)
    evidence_source = "provider" if provider_result and provider_result.get("provider_called") else (
        "cache" if known else "none"
    )
    ages = [float(result["evidence_age_sec"]) for result in known if result.get("evidence_age_sec") is not None]
    return {
        "availability": "unavailable" if all_known else "unknown",
        "evidence_source": evidence_source,
        "evidence_age_sec": max(ages) if ages else None,
        "candidate_count": max((int(result.get("candidate_count") or 0) for result in known), default=0),
        "filters_applied": normalize_provider_filters(filters),
        "materials_checked": len(queries),
        "message": (
            "Cached and provider evidence found no matching YouTube candidates."
            if all_known
            else "Some materials have no fresh search evidence; no more than one provider search was issued."
        ),
    }

@app.get("/api/feed", response_model=FeedResponse)
def feed(
    request: Request,
    material_id: str,
    page: int = 1,
    limit: int = 5,
    autofill: bool = True,
    prefetch: int = 7,
    creative_commons_only: bool = False,
    generation_mode: Literal["slow", "fast"] = "slow",
    min_relevance: float | None = None,
    preferred_video_duration: Literal["any", "short", "medium", "long"] = "any",
    target_clip_duration_sec: int = DEFAULT_TARGET_CLIP_DURATION_SEC,
    target_clip_duration_min_sec: int | None = None,
    target_clip_duration_max_sec: int | None = None,
    exclude_video_ids: str = "",
    exclude_reel_ids: str = "",
    multi_platform_search: bool = False,
):
    from .services.knowledge_level import effective_level_target as _effective_level_target

    _enforce_rate_limit(request, "feed", limit=REELS_GENERATE_RATE_LIMIT_PER_WINDOW)
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
    safe_clip_target, safe_clip_min, safe_clip_max = _resolve_target_clip_duration_bounds(
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
    )
    safe_relevance = _normalize_min_relevance(min_relevance)
    excluded_videos = _parse_excluded_video_ids_param(exclude_video_ids)
    excluded_reels = _parse_excluded_reel_ids_param(exclude_reel_ids)
    with get_conn(transactional=True) as conn:
        material = fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
        if not material:
            raise HTTPException(status_code=404, detail="material_id not found")
        learner_id = _resolve_learner_identity(conn, request)
        progress = reel_service.learner_progress(conn, material_id, learner_id)
        knowledge_level = str(progress.get("selected_level") or "beginner")
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
            knowledge_level=knowledge_level,
            generation_mode=generation_mode,
            creative_commons_only=creative_commons_only,
            source_duration=safe_duration,
            target_clip_duration_sec=safe_clip_target,
            target_clip_duration_min_sec=safe_clip_min,
            target_clip_duration_max_sec=safe_clip_max,
        )
        completed_job = find_completed_generation_job(conn, request_key)
        active_job = find_active_generation_job(conn, request_key)
        cross_request_source = False
        generation_id = str((completed_job or {}).get("result_generation_id") or "") or None
        if generation_id is None:
            head = _fetch_active_generation_row(conn, material_id=material_id, request_key=request_key)
            generation_id = str((head or {}).get("id") or "") or None
        if generation_id is None and active_job:
            generation_id = (
                str(active_job.get("source_generation_id") or "").strip() or None
            )
        if generation_id is None and not completed_job and not active_job:
            generation_id = _verified_cross_request_source_generation(
                conn,
                material_id=material_id,
                learner_id=learner_id,
                request_key=request_key,
                concept_id=None,
            )
            cross_request_source = generation_id is not None
        ranked = _ranked_request_reels(
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
        )
        page_start = (page - 1) * limit
        page_end = page_start + limit
        sparse = len(ranked) < page_end
        unseen_ready = max(0, len(ranked) - page_end)
        material_count = int(
            (
                fetch_one(
                    conn,
                    "SELECT COUNT(*) AS reel_count FROM reels WHERE material_id = ?",
                    (material_id,),
                )
                or {}
            ).get("reel_count")
            or 0
        )
        if (
            autofill
            and (cross_request_source or sparse or unseen_ready <= 4)
            and material_count < MAX_REELS_PER_MATERIAL
        ):
            target_total = min(
                MAX_REELS_PER_MATERIAL,
                max(
                    12,
                    page_end + 4,
                    len(ranked) + max(limit, prefetch),
                ),
            )
            active_job, _created = submit_generation_job(
                conn,
                material_id=material_id,
                concept_id=None,
                request_key=request_key,
                content_fingerprint=content_fingerprint,
                learner_id=learner_id,
                request_params={
                    "material_id": material_id,
                    "concept_id": None,
                    "num_reels": target_total,
                    "exclude_video_ids": excluded_videos,
                    "creative_commons_only": creative_commons_only,
                    "generation_mode": generation_mode,
                    "min_relevance": safe_relevance,
                    "preferred_video_duration": safe_duration,
                    "target_clip_duration_sec": safe_clip_target,
                    "target_clip_duration_min_sec": safe_clip_min,
                    "target_clip_duration_max_sec": safe_clip_max,
                    "knowledge_level": knowledge_level,
                },
                source_generation_id=generation_id,
            )
        reels = ranked[page_start:page_end]
        return {
            "page": page,
            "limit": limit,
            "total": len(ranked),
            "reels": reels,
            "generation_id": generation_id,
            "response_profile": "unified" if generation_id else None,
            "generation_job_id": str((active_job or {}).get("id") or "") or None,
            "generation_job_status": str((active_job or {}).get("status") or "") or None,
            "effective_generation_mode": generation_mode,
            "generation_mode_overridden": False,
            "knowledge_level": knowledge_level,
            "effective_level_target": effective_level,
        }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest):
    _enforce_rate_limit(request, "chat", limit=CHAT_RATE_LIMIT_PER_WINDOW)
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
    with get_conn(transactional=True) as conn:
        learner_id = _resolve_learner_identity(conn, request)
        # Existence check + write under a single transaction so a concurrent
        # delete of the reel can't race us into writing an orphan feedback row.
        exists = fetch_one(
            conn,
            "SELECT id FROM reels WHERE id = ? LIMIT 1",
            (clean_reel_id,),
        )
        if not exists:
            raise HTTPException(status_code=404, detail="reel_id not found")

        reel_service.record_feedback(
            conn,
            reel_id=clean_reel_id,
            helpful=payload.helpful,
            confusing=payload.confusing,
            rating=payload.rating,
            saved=payload.saved,
            learner_id=learner_id,
        )

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
        with get_conn() as conn:
            learner_id = _resolve_learner_identity(conn, request)
        def work(should_cancel: Callable[[], bool]):
            # Keep the monotonic watch-analytics write on a short connection.
            with get_conn() as conn:
                return assessment_service.record_progress(
                    conn,
                    learner_id=learner_id,
                    reel_id=clean_reel_id,
                    max_fraction=payload.max_fraction,
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
        with get_conn() as conn:
            learner_id = _resolve_learner_identity(conn, request)

        def work(should_cancel: Callable[[], bool]):
            with get_conn() as conn:
                return assessment_service.record_scroll(
                    conn,
                    learner_id=learner_id,
                    reel_id=clean_reel_id,
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
        with get_conn() as conn:
            if not fetch_one(conn, "SELECT id FROM materials WHERE id = ?", (payload.material_id,)):
                raise HTTPException(status_code=404, detail="material_id not found")
            learner_id = _resolve_learner_identity(conn, request)
        def work(should_cancel: Callable[[], bool]):
            with get_conn() as conn:
                return assessment_service.next_session(
                    conn,
                    learner_id=learner_id,
                    material_id=payload.material_id,
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
        with get_conn(transactional=True) as conn:
            learner_id = _resolve_learner_identity(conn, request)
            result = assessment_service.answer(
                conn,
                learner_id=learner_id,
                session_id=str(session_id or "").strip(),
                question_id=payload.question_id,
                choice_index=payload.choice_index,
            )
            session = result.get("session") or {}
            if session.get("status") == "completed":
                reel_service.update_level_adjustment(
                    conn, str(session.get("material_id") or ""), learner_id
                )
            return result
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
