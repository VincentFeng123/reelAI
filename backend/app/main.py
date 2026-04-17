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
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parseaddr
from queue import Empty, Queue
from collections.abc import Iterable
from typing import Any, Callable, Literal
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

from .config import get_settings
from .db import (
    DEFAULT_COMMUNITY_VISIBILITY,
    DatabaseIntegrityError,
    LEGACY_COMMUNITY_OWNER_HASH,
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
    MaterialResponse,
    RefinementStatusResponse,
    ReelsCanGenerateAnyRequest,
    ReelsCanGenerateAnyResponse,
    ReelsCanGenerateResponse,
    ReelsGenerateRequest,
    ReelsGenerateResponse,
    AdminSimulationRequest,
    AdminDiagnoseTopicRequest,
    DiagnoseTopicResponse,
    DiagnosticStageResult,
    SimulationResponse,
    SimulationTopicResult,
)
from .services.email import send_welcome_email
from .services.embeddings import EmbeddingService
from .services.liveness import is_video_alive
from .services.material_intelligence import MaterialIntelligenceService
from .services.parsers import ParseError, extract_text_from_file
from .services import progress_bus
from .services.reels import GenerationCancelledError, ReelService
from .services.storage import get_storage
from .services.text_utils import chunk_text, normalize_whitespace
from .services.youtube import YouTubeApiRequestError, YouTubeService, parse_iso8601_duration

from .ingestion.logging_config import (
    publish_progress as _publish_progress_event,
    set_job_id as _set_progress_job_id,
)

from .ingestion.errors import (
    DownloadError as IngestDownloadError,
    IngestError,
    RateLimitedError as IngestRateLimitedError,
    SegmentationError as IngestSegmentationError,
    ServerlessUnavailable as IngestServerlessUnavailable,
    TranscriptionError as IngestTranscriptionError,
    UnsupportedSourceError as IngestUnsupportedSourceError,
)
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

settings = get_settings()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app_instance):
    os.makedirs(settings.data_dir, exist_ok=True)
    init_db()
    _resume_pending_refinement_jobs()
    _warn_if_hosted_auth_email_is_unconfigured()
    yield

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
youtube_service = YouTubeService()
reel_service = ReelService(embedding_service=embedding_service, youtube_service=youtube_service)
SERVERLESS_MODE = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE"))

ingestion_pipeline = IngestionPipeline(
    youtube_service=youtube_service,
    embedding_service=embedding_service,
    settings=settings,
    serverless_mode=SERVERLESS_MODE,
)

REFINEMENT_JOB_WORKERS = 2
_refinement_jobs_lock = threading.Lock()
_refinement_job_executor = None if SERVERLESS_MODE else ThreadPoolExecutor(max_workers=REFINEMENT_JOB_WORKERS)
_scheduled_refinement_job_ids: set[str] = set()
FAST_INITIAL_RESPONSE_REELS = 3
SLOW_INITIAL_RESPONSE_REELS = 5
FAST_MIN_INITIAL_VISIBLE_REELS = 3
SLOW_MIN_INITIAL_VISIBLE_REELS = 5
FAST_MIN_REFINEMENT_TARGET_REELS = 8
SLOW_MIN_REFINEMENT_TARGET_REELS = 10
FAST_REFINEMENT_JOB_BURST_REELS = 8
SLOW_REFINEMENT_JOB_BURST_REELS = 10
REFINEMENT_STAGE_ROOT_EXACT = 0
REFINEMENT_STAGE_ROOT_COMPANION = 1
REFINEMENT_STAGE_MULTI_CLIP_STRICT = 2
REFINEMENT_STAGE_ANCHORED_ADJACENT = 3
REFINEMENT_STAGE_MULTI_CLIP_EXPANDED = 4
REFINEMENT_STAGE_RECOVERY_GRAPH = 5
MAX_REFINEMENT_RECOVERY_STAGE = REFINEMENT_STAGE_RECOVERY_GRAPH
REFINEMENT_JOB_STALE_TIMEOUT_SEC = 300
# Cumulative upper bound on reels stored per material across all feed pages.
# The per-request generation cap in ReelService already bounds each individual
# call, but without this ceiling a client that keeps paginating indefinitely
# can drive total storage to unbounded growth (200 pages * 25 limit = 5,000
# reels per material in the worst case). 300 comfortably covers typical study
# sessions (3-50 reels) while guarding against runaway pagination.
MAX_REELS_PER_MATERIAL = 300

VALID_VIDEO_POOL_MODES = {"short-first", "balanced", "long-form"}
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
DEFAULT_SETTINGS_VIDEO_POOL_MODE = "short-first"
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
REELS_REFINEMENT_STATUS_RATE_LIMIT_PER_WINDOW = 120
FEEDBACK_RATE_LIMIT_PER_WINDOW = 60
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
# Topic-cut runs one Whisper-or-cached transcribe + one chat-completions call
# per video, so it's strictly cheaper than ingest_feed (which loops over many
# items). 6/window matches /api/ingest/url's budget for one full video.
INGEST_TOPIC_CUT_RATE_LIMIT_PER_WINDOW = 6
ADMIN_SIMULATION_RATE_LIMIT_PER_WINDOW = 2
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
            iso_seconds = parse_iso8601_duration(raw)
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
        iso_duration = parse_iso8601_duration(json_ld_duration.group(1))
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
        video_id = youtube_service.extract_video_id_from_url(normalized_source_url)
        if video_id:
            details = youtube_service.video_details([video_id])
            duration = _normalize_duration_seconds((details.get(video_id) or {}).get("duration_sec"))
            if duration is not None:
                return duration

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


def _normalize_video_pool_mode(value: str | None) -> Literal["short-first", "balanced", "long-form"]:
    if value in VALID_VIDEO_POOL_MODES:
        return value
    return "short-first"


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
    generation_mode = generation_mode_raw if generation_mode_raw in {"slow", "fast"} else "fast"
    default_input_mode = _normalize_default_input_mode(str(source.get("default_input_mode") or "").strip().lower() or None)
    min_relevance_threshold = _normalize_settings_min_relevance_threshold(source.get("min_relevance_threshold"))
    start_muted = _normalize_settings_start_muted(source.get("start_muted", 1 if DEFAULT_SETTINGS_START_MUTED else 0))
    video_pool_mode = _normalize_video_pool_mode(str(source.get("video_pool_mode") or "").strip().lower() or None)
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
        video_pool_mode=video_pool_mode,
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


def _build_generation_request_key(
    *,
    material_id: str,
    concept_id: str | None,
    creative_commons_only: bool,
    generation_mode: Literal["slow", "fast"],
    video_pool_mode: Literal["short-first", "balanced", "long-form"],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
) -> str:
    payload = {
        "material_id": material_id,
        "concept_id": concept_id or "",
        "creative_commons_only": bool(creative_commons_only),
        "generation_mode": generation_mode,
        "video_pool_mode": video_pool_mode,
        "preferred_video_duration": preferred_video_duration,
        "target_clip_duration_sec": int(target_clip_duration_sec),
        "target_clip_duration_min_sec": int(target_clip_duration_min_sec or 0),
        "target_clip_duration_max_sec": int(target_clip_duration_max_sec or 0),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
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


def _request_curated_companion_terms(subject_tag: str | None) -> list[str]:
    cleaned = str(subject_tag or "").strip()
    if not cleaned:
        return []

    normalized_subject = reel_service._normalize_query_key(cleaned)
    ordered: list[str] = []
    seen: set[str] = set()

    def add_term(raw_value: str) -> None:
        candidate = str(raw_value or "").strip()
        normalized = reel_service._normalize_query_key(candidate)
        if not candidate or not normalized or normalized == normalized_subject or normalized in seen:
            return
        seen.add(normalized)
        ordered.append(candidate)

    for mapping in (
        getattr(reel_service.topic_expansion_service, "STATIC_TOPIC_SUBTOPICS", {}),
        getattr(reel_service, "BROAD_TOPIC_SUBTOPICS", {}),
    ):
        for raw_topic, topic_terms in mapping.items():
            if reel_service._normalize_query_key(raw_topic) != normalized_subject:
                continue
            for term in topic_terms:
                add_term(str(term or ""))
            break

    for term in reel_service._expand_controlled_synonyms([cleaned], fast_mode=True):
        normalized_term = reel_service._normalize_query_key(str(term or ""))
        if len(normalized_term.split()) < 2:
            continue
        add_term(str(term or ""))
    return ordered[:10]


def _request_root_anchor_terms(subject_tag: str | None) -> tuple[list[str], list[str], list[str]]:
    cleaned = str(subject_tag or "").strip()
    if not cleaned:
        return ([], [], [])
    aliases = reel_service.topic_expansion_service._deterministic_alias_terms(topic=cleaned, canonical_topic=cleaned)
    companions = reel_service.topic_expansion_service._deterministic_companion_terms(topic=cleaned, canonical_topic=cleaned)
    page_one_companions = _request_curated_companion_terms(cleaned)
    deduped_root = _dedupe_request_terms([cleaned, *aliases])
    deduped_companions = _dedupe_request_terms([*companions, *page_one_companions])
    return deduped_root, deduped_companions, page_one_companions


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

    if len(shaped) < target_visible_count and strict_topic_only and subject_tag:
        relaxed_topic_shaped = _shape_reels_for_request_context(
            _filter_reels_by_min_relevance(ranked, min_relevance),
            page=page,
            limit=limit,
            subject_tag=subject_tag,
            strict_topic_only=False,
        )
        shaped = _merge_request_reel_lists(shaped, relaxed_topic_shaped)

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
            strict_topic_only=False,
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
            if page <= 1:
                if not has_root_anchor and not (
                    has_page_one_companion_anchor and (educational_support or topical_short_support)
                ):
                    continue
            elif not has_root_anchor and not (has_companion_anchor and (educational_support or topical_short_support)):
                continue

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
    row = fetch_one(conn, "SELECT COUNT(*) AS reel_count FROM reels WHERE generation_id = ?", (generation_id,))
    return max(0, int((row or {}).get("reel_count") or 0))


def _fetch_generation_row(conn, generation_id: str | None) -> dict[str, Any] | None:
    if not generation_id:
        return None
    return fetch_one(conn, "SELECT * FROM reel_generations WHERE id = ?", (generation_id,))


def _normalize_provider_stage(value: object) -> Literal["youtube", "youtube_graph", "youtube_external", "dailymotion", "wikimedia", "stock"]:
    clean = str(value or "").strip().lower()
    if clean in {"youtube", "youtube_graph", "youtube_external", "dailymotion", "wikimedia", "stock"}:
        return clean  # type: ignore[return-value]
    if clean in {"wikimedia_commons", "internet_archive", "pexels_video", "pixabay_video"}:
        return "stock" if clean != "wikimedia_commons" else "wikimedia"
    return "youtube"


def _next_provider_stage(current_stage: object) -> Literal["dailymotion", "wikimedia", "stock"] | None:
    current = _normalize_provider_stage(current_stage)
    if current in {"youtube", "youtube_graph", "youtube_external"}:
        return "dailymotion"
    if current == "dailymotion":
        return "wikimedia"
    if current == "wikimedia":
        return "stock"
    return None


def _search_status_provider_stage(
    provider_stage: object,
    *,
    recovery_stage: int | None = None,
) -> Literal["youtube", "youtube_graph", "youtube_external", "dailymotion", "wikimedia", "stock"]:
    normalized = _normalize_provider_stage(provider_stage)
    if normalized != "youtube":
        return normalized
    safe_recovery_stage = max(0, int(recovery_stage or 0))
    if safe_recovery_stage > MAX_REFINEMENT_RECOVERY_STAGE:
        return "youtube_external"
    if safe_recovery_stage >= REFINEMENT_STAGE_RECOVERY_GRAPH:
        return "youtube_graph"
    return "youtube"


def _fetch_generation_head_row(conn, *, material_id: str, request_key: str) -> dict[str, Any] | None:
    return fetch_one(
        conn,
        "SELECT * FROM reel_generation_heads WHERE material_id = ? AND request_key = ?",
        (material_id, request_key),
    )


def _generation_head_row(
    existing_row: dict[str, Any] | None,
    *,
    material_id: str,
    request_key: str,
) -> dict[str, Any]:
    return {
        "id": str((existing_row or {}).get("id") or _build_generation_head_id(material_id, request_key)),
        "material_id": material_id,
        "request_key": request_key,
        "active_generation_id": str((existing_row or {}).get("active_generation_id") or ""),
        "search_state": str((existing_row or {}).get("search_state") or "idle"),
        "can_request_more": 1 if bool(_to_int((existing_row or {}).get("can_request_more"), 1)) else 0,
        "exhausted_reason": str((existing_row or {}).get("exhausted_reason") or "") or None,
        "provider_stage": _normalize_provider_stage((existing_row or {}).get("provider_stage")),
        "provider_fallback_enabled": 1 if bool(_to_int((existing_row or {}).get("provider_fallback_enabled"), 0)) else 0,
        "recovery_stage": max(0, _to_int((existing_row or {}).get("recovery_stage"), 0)),
        "stage_exhausted_json": str((existing_row or {}).get("stage_exhausted_json") or "{}"),
        "last_progress_at": str((existing_row or {}).get("last_progress_at") or "") or None,
        "last_job_id": str((existing_row or {}).get("last_job_id") or "") or None,
        "search_session_version": max(1, _to_int((existing_row or {}).get("search_session_version"), 1)),
        "updated_at": str((existing_row or {}).get("updated_at") or now_iso()),
    }


def _upsert_generation_head(
    conn,
    *,
    material_id: str,
    request_key: str,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = _generation_head_row(
        _fetch_generation_head_row(conn, material_id=material_id, request_key=request_key),
        material_id=material_id,
        request_key=request_key,
    )
    for key, value in (updates or {}).items():
        if key == "provider_stage":
            row[key] = _normalize_provider_stage(value)
        elif key == "recovery_stage":
            row[key] = max(0, _to_int(value, row["recovery_stage"]))
        elif key == "search_session_version":
            row[key] = max(1, _to_int(value, row["search_session_version"]))
        elif key in {"can_request_more", "provider_fallback_enabled"}:
            row[key] = 1 if bool(value) else 0
        elif key == "stage_exhausted_json":
            if isinstance(value, str):
                row[key] = value or "{}"
            else:
                row[key] = dumps_json(_normalize_stage_exhausted(value))
        elif key == "updated_at":
            row[key] = str(value or row[key])
        else:
            row[key] = value
    row["updated_at"] = str(row.get("updated_at") or now_iso())
    if not str(row.get("stage_exhausted_json") or "").strip():
        row["stage_exhausted_json"] = "{}"
    try:
        upsert(conn, "reel_generation_heads", row)
    except (sqlite3.OperationalError, Exception) as exc:
        if "no column named id" not in str(exc).lower() and "column" not in str(exc).lower():
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
            (
                material_id,
                request_key,
                str(row.get("active_generation_id") or ""),
                str(row.get("updated_at") or now_iso()),
            ),
        )
    return row


def _head_stage_exhausted(head_row: dict[str, Any] | None) -> dict[str, bool]:
    if not head_row:
        return {}
    raw_value = head_row.get("stage_exhausted_json")
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value or "{}")
        except json.JSONDecodeError:
            parsed = {}
        return _normalize_stage_exhausted(parsed)
    return _normalize_stage_exhausted(raw_value)


def _head_search_session_version(head_row: dict[str, Any] | None) -> int:
    return max(1, _to_int((head_row or {}).get("search_session_version"), 1))


def _search_status_from_head(
    head_row: dict[str, Any] | None,
    *,
    job_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not head_row:
        return {
            "state": "idle",
            "can_request_more": True,
            "reason": None,
            "provider_stage": "youtube",
            "recovery_stage": 0,
            "last_progress_at": None,
        }
    state = str(head_row.get("search_state") or "idle").strip().lower()
    can_request_more = bool(_to_int(head_row.get("can_request_more"), 1))
    reason = str(head_row.get("exhausted_reason") or "").strip() or None
    recovery_stage = max(0, _to_int(head_row.get("recovery_stage"), 0))
    provider_stage = _search_status_provider_stage(head_row.get("provider_stage"), recovery_stage=recovery_stage)
    last_progress_at = _normalize_datetime_for_api(head_row.get("last_progress_at"))
    job_status = str((job_row or {}).get("status") or "").strip().lower()
    if state not in {"stalled", "exhausted", "failed"} and job_status in {"queued", "running"}:
        state = "refining" if job_status == "running" else "active"
    if state not in {"idle", "active", "refining", "stalled", "exhausted", "failed"}:
        state = "idle"
    return {
        "state": state,
        "can_request_more": can_request_more,
        "reason": reason,
        "provider_stage": provider_stage,
        "recovery_stage": recovery_stage,
        "last_progress_at": last_progress_at,
    }


def _job_activity_started_at(job_row: dict[str, Any] | None) -> datetime | None:
    if not job_row:
        return None
    return _parse_optional_iso_datetime(job_row.get("started_at")) or _parse_optional_iso_datetime(job_row.get("created_at"))


def _refinement_job_is_stale(job_row: dict[str, Any] | None) -> bool:
    started_at = _job_activity_started_at(job_row)
    if started_at is None:
        return False
    age_sec = (datetime.now(timezone.utc) - started_at).total_seconds()
    return age_sec >= REFINEMENT_JOB_STALE_TIMEOUT_SEC


def _update_head_for_search_state(
    conn,
    *,
    material_id: str,
    request_key: str,
    search_state: str,
    can_request_more: bool,
    exhausted_reason: str | None = None,
    provider_stage: object | None = None,
    recovery_stage: int | None = None,
    stage_exhausted: dict[str, bool] | None = None,
    last_job_id: str | None = None,
    last_progress_at: str | None = None,
    provider_fallback_enabled: bool | None = None,
    search_session_version: int | None = None,
    active_generation_id: str | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "search_state": search_state,
        "can_request_more": can_request_more,
        "exhausted_reason": exhausted_reason,
    }
    if provider_stage is not None:
        updates["provider_stage"] = provider_stage
    if recovery_stage is not None:
        updates["recovery_stage"] = recovery_stage
    if stage_exhausted is not None:
        updates["stage_exhausted_json"] = stage_exhausted
    if last_job_id is not None:
        updates["last_job_id"] = last_job_id
    if last_progress_at is not None:
        updates["last_progress_at"] = last_progress_at
    if provider_fallback_enabled is not None:
        updates["provider_fallback_enabled"] = provider_fallback_enabled
    if search_session_version is not None:
        updates["search_session_version"] = search_session_version
    if active_generation_id is not None:
        updates["active_generation_id"] = active_generation_id
    return _upsert_generation_head(
        conn,
        material_id=material_id,
        request_key=request_key,
        updates=updates,
    )


def _fetch_active_generation_row(conn, *, material_id: str, request_key: str) -> dict[str, Any] | None:
    head = _fetch_generation_head_row(conn, material_id=material_id, request_key=request_key)
    if not head:
        return None
    return _fetch_generation_row(conn, str(head.get("active_generation_id") or ""))


def _fetch_refinement_job_for_generation(conn, source_generation_id: str | None) -> dict[str, Any] | None:
    if not source_generation_id:
        return None
    job_row = fetch_one(
        conn,
        """
        SELECT *
        FROM reel_generation_jobs
        WHERE source_generation_id = ?
          AND status IN ('queued', 'running')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (source_generation_id,),
    )
    if not job_row:
        return None
    if str(job_row.get("status") or "").strip().lower() == "running" and _refinement_job_is_stale(job_row):
        job_id = str(job_row.get("id") or "").strip()
        logger.warning("Resetting stale refinement job %s to failed", job_id)
        execute_modify(
            conn,
            "UPDATE reel_generation_jobs SET status = 'failed', error_text = 'stale (timed out)', completed_at = ? WHERE id = ?",
            (now_iso(), job_id),
        )
        return None
    if str(job_row.get("status") or "").strip().lower() == "queued":
        resumed = _resume_queued_refinement_job(conn, job_row)
        if resumed is None:
            return None
        job_row = resumed
    return job_row


def _submit_refinement_job(job_id: str) -> bool:
    clean_job_id = str(job_id or "").strip()
    if not clean_job_id or SERVERLESS_MODE:
        return False
    if _refinement_job_executor is None:
        return False
    with _refinement_jobs_lock:
        if clean_job_id in _scheduled_refinement_job_ids:
            return False
        _scheduled_refinement_job_ids.add(clean_job_id)
        try:
            _refinement_job_executor.submit(_run_refinement_job_with_cleanup, clean_job_id)
        except Exception:
            _scheduled_refinement_job_ids.discard(clean_job_id)
            raise
    return True


def _run_refinement_job_with_cleanup(job_id: str) -> None:
    """
    Instrumented wrapper around `_run_refinement_job` (Phase D.1 / D.2).

    Binds the current thread's `job_id` contextvar so any `publish_progress(...)`
    calls inside the worker auto-route onto the progress bus topic keyed on `job_id`.
    Emits terminal `job_start` / `job_end` events and always calls
    `progress_bus.terminate()` so attached NDJSON subscribers get a clean EOF.
    """
    clean_id = str(job_id or "").strip()
    _set_progress_job_id(clean_id)
    started_mono = time.monotonic()
    _publish_progress_event("job_start", job_id=clean_id)
    exc_class: str | None = None
    try:
        _run_refinement_job(job_id)
    except BaseException as exc:
        exc_class = type(exc).__name__
        raise
    finally:
        duration_ms = int((time.monotonic() - started_mono) * 1000)
        outcome = "ok" if exc_class is None else "error"
        _publish_progress_event(
            "job_end",
            job_id=clean_id,
            duration_ms=duration_ms,
            outcome=outcome,
            error_class=exc_class,
        )
        progress_bus.terminate(
            clean_id,
            final_event={
                "event": "job_terminal",
                "job_id": clean_id,
                "outcome": outcome,
                "error_class": exc_class,
                "duration_ms": duration_ms,
            },
        )
        with _refinement_jobs_lock:
            _scheduled_refinement_job_ids.discard(clean_id)
        _set_progress_job_id(None)


def _resume_queued_refinement_job(conn, job_row: dict[str, Any]) -> dict[str, Any] | None:
    job_id = str(job_row.get("id") or "").strip()
    if not job_id or SERVERLESS_MODE or _refinement_job_executor is None:
        return job_row

    source_generation_id = str(job_row.get("source_generation_id") or "").strip()
    source_generation = _fetch_generation_row(conn, source_generation_id)
    if not source_generation:
        upsert(
            conn,
            "reel_generation_jobs",
            {
                **job_row,
                "status": "failed",
                "completed_at": now_iso(),
                "error_text": "source_generation_missing",
            },
        )
        return None

    active_generation = _fetch_active_generation_row(
        conn,
        material_id=str(job_row.get("material_id") or ""),
        request_key=str(job_row.get("request_key") or ""),
    )
    active_generation_id = str((active_generation or {}).get("id") or "").strip()
    if active_generation_id != source_generation_id:
        upsert(
            conn,
            "reel_generation_jobs",
            {
                **job_row,
                "status": "superseded",
                "completed_at": now_iso(),
                "error_text": None,
            },
        )
        return None

    with _refinement_jobs_lock:
        _scheduled_refinement_job_ids.discard(job_id)
    _submit_refinement_job(job_id)
    return job_row


def _resume_pending_refinement_jobs() -> None:
    if SERVERLESS_MODE or _refinement_job_executor is None:
        return

    job_ids_to_resume: list[str] = []
    with get_conn(transactional=True) as conn:
        pending_jobs = fetch_all(
            conn,
            """
            SELECT *
            FROM reel_generation_jobs
            WHERE status IN ('queued', 'running')
            ORDER BY created_at ASC, id ASC
            """,
        )
        for job_row in pending_jobs:
            job_id = str(job_row.get("id") or "").strip()
            if not job_id:
                continue

            source_generation_id = str(job_row.get("source_generation_id") or "").strip()
            source_generation = _fetch_generation_row(conn, source_generation_id)
            if not source_generation:
                upsert(
                    conn,
                    "reel_generation_jobs",
                    {
                        **job_row,
                        "status": "failed",
                        "completed_at": now_iso(),
                        "error_text": "source_generation_missing",
                    },
                )
                continue

            active_generation = _fetch_active_generation_row(
                conn,
                material_id=str(job_row.get("material_id") or ""),
                request_key=str(job_row.get("request_key") or ""),
            )
            active_generation_id = str((active_generation or {}).get("id") or "").strip()
            if active_generation_id != source_generation_id:
                upsert(
                    conn,
                    "reel_generation_jobs",
                    {
                        **job_row,
                        "status": "superseded",
                        "completed_at": now_iso(),
                        "error_text": None,
                    },
                )
                continue

            if str(job_row.get("status") or "").strip().lower() == "running":
                upsert(
                    conn,
                    "reel_generation_jobs",
                    {
                        **job_row,
                        "status": "queued",
                        "started_at": None,
                        "completed_at": None,
                        "error_text": None,
                    },
                )
            job_ids_to_resume.append(job_id)

    for job_id in job_ids_to_resume:
        _submit_refinement_job(job_id)


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
            "generation_mode": str(row.get("generation_mode") or "fast"),
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
        parsed = round(float(value or 0.0), 2)
    except (TypeError, ValueError):
        parsed = 0.0
    return f"{parsed:.2f}"


def _reel_identity_key(reel: dict[str, Any]) -> tuple[str, str]:
    reel_id = str(reel.get("reel_id") or "").strip()
    video_url = str(reel.get("video_url") or "").strip()
    embed_match = re.search(r"/embed/([^?&/]+)", video_url)
    if embed_match:
        video_identity = embed_match.group(1)
    else:
        parsed = urlparse(video_url)
        video_identity = parsed.path or video_url
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

    if not merged:
        return merged

    # Cross-generation regrouping: each batch is already video-grouped by
    # ``ranked_feed``, but concatenating batches can split a video's reels
    # across generation boundaries (gen1's A-reels, other videos, then
    # gen2's A-reels later). Re-group by video so same-clip reels play
    # contiguously in chronological order — exact same behavior as
    # ``ranked_feed`` produces within a single generation. This keeps the
    # iOS and web feeds aligned with the within-generation grouping.
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_first_rank: dict[str, int] = {}
    group_best_score: dict[str, float] = {}
    group_best_created_at: dict[str, str] = {}
    unkeyed: list[dict[str, Any]] = []
    for idx, reel in enumerate(merged):
        _, clip_key = _reel_identity_key(reel)
        group_key = clip_key.split(":", 1)[0] if clip_key else ""
        if not group_key:
            # Defensive: reels without an identifiable video identity stay
            # in their merged-order position via a stable rank sentinel.
            unkeyed.append(reel)
            continue
        if group_key not in grouped:
            grouped[group_key] = []
            group_first_rank[group_key] = idx
        grouped[group_key].append(reel)
        try:
            score = float(reel.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > group_best_score.get(group_key, float("-inf")):
            group_best_score[group_key] = score
            group_best_created_at[group_key] = str(reel.get("created_at") or "")

    # Order groups by best score (desc), then by first-merged-rank (asc)
    # as a deterministic tiebreak — preserving the original cross-batch
    # ordering when scores are tied or absent.
    ordered_groups = sorted(
        grouped.keys(),
        key=lambda k: (
            -group_best_score.get(k, 0.0),
            group_first_rank.get(k, 0),
        ),
    )
    regrouped: list[dict[str, Any]] = []
    for group_key in ordered_groups:
        # Within each group, play reels in chronological order so a
        # multi-part clip's narrative (part 1 → 2 → 3) stays intact.
        group_reels = sorted(
            grouped[group_key],
            key=lambda r: (
                float(r.get("t_start") or 0.0),
                float(r.get("t_end") or 0.0),
            ),
        )
        regrouped.extend(group_reels)
    # Append any reels we couldn't group (missing video identity) at the
    # tail, preserving their merged order.
    regrouped.extend(unkeyed)
    return regrouped


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


def _extend_active_generation(
    conn,
    *,
    material_id: str,
    concept_id: str | None,
    request_key: str,
    required_count: int,
    active_generation_id: str,
    active_response_profile: str | None,
    generation_mode: Literal["slow", "fast"],
    creative_commons_only: bool,
    min_relevance: float | None,
    video_pool_mode: Literal["short-first", "balanced", "long-form"],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
    exclude_video_ids: list[str] | None = None,
    page_hint: int = 1,
    on_reel_created: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if should_cancel is not None and should_cancel():
        raise GenerationCancelledError("Generation cancelled.")
    fast_mode = generation_mode == "fast"
    prior_generation_ids = _response_generation_ids(conn, active_generation_id)
    next_generation_id = _create_generation_row(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        request_key=request_key,
        generation_mode=generation_mode,
        retrieval_profile="deep",
        source_generation_id=active_generation_id,
    )

    reel_service.generate_reels(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        num_reels=max(1, int(required_count)),
        creative_commons_only=creative_commons_only,
        exclude_video_ids=exclude_video_ids,
        fast_mode=fast_mode,
        video_pool_mode=video_pool_mode,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        retrieval_profile="deep",
        generation_id=next_generation_id,
        exclude_generation_ids=prior_generation_ids,
        min_relevance_threshold=float(min_relevance or 0.0),
        page_hint=page_hint,
        recovery_stage=_initial_recovery_stage(page_hint=page_hint),
        on_reel_created=on_reel_created,
        should_cancel=should_cancel,
    )

    if _count_generation_reels(conn, next_generation_id) <= 0:
        _complete_generation(
            conn,
            generation_id=next_generation_id,
            retrieval_profile="deep",
            status="failed",
            error_text="no_reels_generated",
        )
        fallback_reels = _ranked_request_reels(
            conn,
            material_id=material_id,
            fast_mode=fast_mode,
            generation_id=active_generation_id,
            min_relevance=min_relevance,
            preferred_video_duration=preferred_video_duration,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
            exclude_video_ids=exclude_video_ids,
            page=page_hint,
            limit=max(5, min(25, required_count)),
        )
        response_payload = _build_generation_response_payload(
            reels=fallback_reels,
            generation_id=active_generation_id,
            response_profile=active_response_profile,
            job_row=None,
        )
        _log_generation_response_summary(
            event="extend_active_generation_empty",
            material_id=material_id,
            request_key=request_key,
            generation_mode=generation_mode,
            required_count=required_count,
            response_payload=response_payload,
            page_hint=page_hint,
        )
        return {**response_payload, "request_key": request_key}

    _activate_generation(
        conn,
        material_id=material_id,
        request_key=request_key,
        generation_id=next_generation_id,
        retrieval_profile="deep",
    )
    expanded_reels = _ranked_request_reels(
        conn,
        material_id=material_id,
        fast_mode=fast_mode,
        generation_id=next_generation_id,
        min_relevance=min_relevance,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        exclude_video_ids=exclude_video_ids,
        page=page_hint,
        limit=max(5, min(25, required_count)),
    )
    job_row = _queue_refinement_if_needed(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        request_key=request_key,
        generation_id=next_generation_id,
        current_reels=expanded_reels,
        required_count=required_count,
        generation_mode=generation_mode,
        creative_commons_only=creative_commons_only,
        preferred_video_duration=preferred_video_duration,
        video_pool_mode=video_pool_mode,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        min_relevance=min_relevance,
        page_hint=page_hint,
        page_size_hint=max(5, min(25, required_count)),
    )
    response_payload = _build_generation_response_payload(
        reels=expanded_reels,
        generation_id=next_generation_id,
        response_profile="deep",
        job_row=job_row,
    )
    _log_generation_response_summary(
        event="extend_active_generation_complete",
        material_id=material_id,
        request_key=request_key,
        generation_mode=generation_mode,
        required_count=required_count,
        response_payload=response_payload,
        page_hint=page_hint,
    )
    return {**response_payload, "request_key": request_key}


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
    generation_ids = _response_generation_ids(conn, generation_id)
    if generation_ids:
        ranked_batches = [
            reel_service.ranked_feed(
                conn,
                material_id,
                fast_mode=fast_mode,
                generation_id=current_generation_id,
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
        )
    excluded_video_id_set = {
        str(video_id or "").strip()
        for video_id in (exclude_video_ids or [])
        if str(video_id or "").strip()
    }
    if excluded_video_id_set:
        ranked = [
            reel for reel in ranked
            if str(reel.get("video_id") or "").strip() not in excluded_video_id_set
        ]

    unique_vids = {
        str(reel.get("video_id") or "").strip()
        for reel in ranked
        if str(reel.get("video_id") or "").strip()
    }
    if unique_vids:
        alive_map: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(unique_vids))) as pool:
            futures = {pool.submit(is_video_alive, vid, conn=conn): vid for vid in unique_vids}
            for future in futures:
                vid = futures[future]
                try:
                    alive_map[vid] = bool(future.result())
                except Exception:
                    alive_map[vid] = True  # fail open so a probe error never hides a reel
        ranked = [
            reel for reel in ranked
            if alive_map.get(str(reel.get("video_id") or "").strip(), True)
        ]

    if page <= 1:
        return _shape_request_page_reels(
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
    return _merge_request_reel_lists(*cumulative_batches)


def _build_generation_response_payload(
    *,
    reels: list[dict[str, Any]],
    generation_id: str | None,
    response_profile: str | None,
    job_row: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "reels": reels,
        "generation_id": generation_id,
        "response_profile": response_profile,
        "refinement_job_id": str((job_row or {}).get("id") or "") or None,
        "refinement_status": str((job_row or {}).get("status") or "") or None,
    }


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


def _short_debug_token(value: str | None, *, limit: int = 12) -> str | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit]}..."


def _generation_source_video_count(reels: list[dict[str, Any]]) -> int:
    source_ids: set[str] = set()
    for reel in reels:
        source_id = str(reel.get("video_id") or reel.get("video_url") or "").strip()
        if source_id:
            source_ids.add(source_id)
    return len(source_ids)


def _log_generation_response_summary(
    *,
    event: str,
    material_id: str,
    request_key: str | None,
    generation_mode: str,
    required_count: int,
    response_payload: dict[str, Any],
    page_hint: int = 1,
) -> None:
    reels = list(response_payload.get("reels") or [])
    payload = {
        "event": event,
        "material_id": material_id,
        "request_key": _short_debug_token(request_key),
        "generation_id": _short_debug_token(str(response_payload.get("generation_id") or "") or None),
        "generation_mode": generation_mode,
        "response_profile": str(response_payload.get("response_profile") or "") or None,
        "required_count": int(max(1, required_count)),
        "returned_count": len(reels),
        "unique_source_videos": _generation_source_video_count(reels),
        "page_hint": int(max(1, page_hint)),
        "refinement_status": str(response_payload.get("refinement_status") or "") or None,
    }
    if len(reels) == 0:
        logger.warning("Generation response summary: %s", json.dumps(payload, sort_keys=True))
        return
    if len(reels) < max(1, int(required_count or 1)) or settings.retrieval_debug_logging:
        logger.info("Generation response summary: %s", json.dumps(payload, sort_keys=True))


def _initial_response_reel_target(
    *,
    required_count: int,
    generation_mode: Literal["slow", "fast"],
    sync_deep_fallback: Literal["always", "if_empty", "never"],
) -> int:
    safe_required = max(1, int(required_count))
    if sync_deep_fallback == "always":
        return safe_required
    if safe_required <= 2:
        return safe_required
    cap = FAST_INITIAL_RESPONSE_REELS if generation_mode == "fast" else SLOW_INITIAL_RESPONSE_REELS
    return max(2, min(safe_required, cap))


def _minimum_initial_response_reels(
    *,
    required_count: int,
    generation_mode: Literal["slow", "fast"],
    sync_deep_fallback: Literal["always", "if_empty", "never"],
) -> int:
    if sync_deep_fallback == "never":
        return 1
    safe_required = max(1, int(required_count))
    floor = FAST_MIN_INITIAL_VISIBLE_REELS if generation_mode == "fast" else SLOW_MIN_INITIAL_VISIBLE_REELS
    return max(1, min(safe_required, floor))


def _refinement_target_reel_count(
    *,
    required_count: int,
    generation_mode: Literal["slow", "fast"],
    target_page: int = 1,
    page_size_hint: int = 5,
    existing_source_count: int = 0,
) -> int:
    safe_existing = max(0, int(existing_source_count))
    inventory_target = _refinement_inventory_target_count(
        required_count=required_count,
        generation_mode=generation_mode,
        target_page=target_page,
        page_size_hint=page_size_hint,
        existing_source_count=safe_existing,
    )
    remaining_gap = max(0, inventory_target - safe_existing)
    if remaining_gap <= 0:
        return 0
    burst = FAST_REFINEMENT_JOB_BURST_REELS if generation_mode == "fast" else SLOW_REFINEMENT_JOB_BURST_REELS
    minimum_seed = FAST_MIN_REFINEMENT_TARGET_REELS if generation_mode == "fast" else SLOW_MIN_REFINEMENT_TARGET_REELS
    minimum_target = minimum_seed if safe_existing <= 0 else 1
    return max(minimum_target, min(remaining_gap, burst))


def _refinement_inventory_target_count(
    *,
    required_count: int,
    generation_mode: Literal["slow", "fast"],
    target_page: int = 1,
    page_size_hint: int = 5,
    existing_source_count: int = 0,
) -> int:
    safe_required = max(1, int(required_count))
    safe_target_page = max(1, int(target_page or 1))
    safe_page_size = max(1, int(page_size_hint or 1))
    safe_existing = max(0, int(existing_source_count))
    min_target = FAST_MIN_REFINEMENT_TARGET_REELS if generation_mode == "fast" else SLOW_MIN_REFINEMENT_TARGET_REELS
    return max(min_target, safe_required, safe_target_page * safe_page_size, safe_existing)


def _refinement_horizon_page(
    *,
    required_count: int,
    generation_mode: Literal["slow", "fast"],
    target_page: int = 1,
    page_size_hint: int = 5,
    existing_source_count: int = 0,
) -> int:
    safe_required = max(1, int(required_count or 1))
    safe_target_page = max(1, int(target_page or 1))
    safe_page_size = max(1, int(page_size_hint or 1))
    safe_existing = max(0, int(existing_source_count or 0))
    horizon_count = max(safe_required, safe_existing)
    return max(safe_target_page, int(math.ceil(horizon_count / safe_page_size)))


def _initial_recovery_stage(*, page_hint: int) -> int:
    return 0


def _normalize_stage_exhausted(raw_value: object) -> dict[str, bool]:
    if isinstance(raw_value, dict):
        return {str(key): bool(value) for key, value in raw_value.items()}
    if isinstance(raw_value, list):
        return {str(item): True for item in raw_value}
    return {}


def _first_unexhausted_recovery_stage(stage_exhausted: dict[str, bool] | None) -> int:
    exhausted = stage_exhausted or {}
    for stage in range(0, MAX_REFINEMENT_RECOVERY_STAGE + 1):
        if not bool(exhausted.get(str(stage))):
            return stage
    return MAX_REFINEMENT_RECOVERY_STAGE + 1


def _refinement_stage_kind(recovery_stage: int) -> str:
    safe_stage = int(recovery_stage or 0)
    return (
        "window"
        if safe_stage in {REFINEMENT_STAGE_MULTI_CLIP_STRICT, REFINEMENT_STAGE_MULTI_CLIP_EXPANDED}
        else "source"
    )


def _visible_reel_clip_keys(reels: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for reel in reels:
        _reel_id, clip_key = _reel_identity_key(reel)
        if clip_key:
            keys.add(clip_key)
    return keys


def _infer_frontier_family_from_reel(reel: dict[str, Any]) -> str:
    retrieval_stage = str(reel.get("retrieval_stage") or "").strip().lower()
    query_strategy = str(reel.get("query_strategy") or "").strip().lower()
    source_surface = str(reel.get("source_surface") or "").strip().lower()
    if retrieval_stage == "high_precision" and query_strategy == "literal":
        return "root_exact"
    if source_surface in {
        "youtube_channel",
        "youtube_related",
        "duckduckgo_site",
        "duckduckgo_quoted",
        "bing_site",
        "bing_quoted",
        "local_cache",
    }:
        return "recovery_graph"
    if retrieval_stage == "recovery" or query_strategy == "recovery_adjacent":
        return "anchored_adjacent"
    return "root_companion"


def _update_frontier_visible_growth(
    conn,
    *,
    material_id: str,
    request_key: str,
    reels: list[dict[str, Any]],
) -> None:
    if not material_id or not request_key or not reels:
        return
    visible_by_family: dict[str, int] = {}
    for reel in reels:
        family = _infer_frontier_family_from_reel(reel)
        visible_by_family[family] = visible_by_family.get(family, 0) + 1
    updated_at = now_iso()
    for family, count in visible_by_family.items():
        if count <= 0:
            continue
        entry = fetch_one(
            conn,
            """
            SELECT id, new_visible_reels
            FROM request_frontier_entries
            WHERE material_id = ?
              AND request_key = ?
              AND family_key LIKE ?
              AND exhausted = 0
            ORDER BY COALESCE(last_run_at, updated_at) DESC, updated_at DESC
            LIMIT 1
            """,
            (material_id, request_key, f"{family}:%"),
        )
        if not entry:
            entry = fetch_one(
                conn,
                """
                SELECT id, new_visible_reels
                FROM request_frontier_entries
                WHERE material_id = ?
                  AND request_key = ?
                  AND family_key LIKE ?
                ORDER BY COALESCE(last_run_at, updated_at) DESC, updated_at DESC
                LIMIT 1
                """,
                (material_id, request_key, f"{family}:%"),
            )
        if not entry:
            continue
        execute_modify(
            conn,
            """
            UPDATE request_frontier_entries
            SET new_visible_reels = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (max(0, int(entry.get("new_visible_reels") or 0)) + count, updated_at, str(entry.get("id") or "")),
        )


def _count_generation_unique_videos(conn, generation_id: str) -> int:
    row = fetch_one(
        conn,
        "SELECT COUNT(DISTINCT video_id) AS video_count FROM reels WHERE generation_id = ?",
        (generation_id,),
    )
    return max(0, int((row or {}).get("video_count") or 0))


def _advance_refinement_state(
    *,
    recovery_stage: int,
    stage_exhausted: dict[str, bool],
    no_new_sources_passes: int,
    no_new_windows_passes: int,
    no_new_visible_reels_passes: int,
    new_unique_videos: int,
    new_unique_clip_windows: int,
    new_unique_visible_reels: int,
) -> dict[str, Any]:
    safe_stage = max(0, int(recovery_stage or 0))
    stage_kind = _refinement_stage_kind(safe_stage)
    updated_stage_exhausted = dict(stage_exhausted)
    next_no_new_sources_passes = 0 if new_unique_videos > 0 else no_new_sources_passes + 1
    next_no_new_windows_passes = 0 if new_unique_clip_windows > 0 else no_new_windows_passes + 1
    next_no_new_visible_reels_passes = 0 if new_unique_visible_reels > 0 else no_new_visible_reels_passes + 1
    if stage_kind == "window":
        stage_is_exhausted = next_no_new_windows_passes >= 2 and next_no_new_visible_reels_passes >= 2
    else:
        stage_is_exhausted = next_no_new_sources_passes >= 2 and next_no_new_visible_reels_passes >= 2
    if stage_is_exhausted:
        updated_stage_exhausted[str(safe_stage)] = True
    next_stage = safe_stage
    if stage_is_exhausted:
        next_stage = _first_unexhausted_recovery_stage(updated_stage_exhausted)
        if next_stage != safe_stage:
            next_no_new_sources_passes = 0
            next_no_new_windows_passes = 0
            next_no_new_visible_reels_passes = 0
    if new_unique_visible_reels > 0:
        growth_reason = "new_visible_reels"
    elif new_unique_clip_windows > 0:
        growth_reason = "new_clip_windows"
    elif new_unique_videos > 0:
        growth_reason = "new_source_videos"
    elif stage_is_exhausted:
        growth_reason = "stage_exhausted"
    else:
        growth_reason = "no_growth"
    return {
        "next_stage": next_stage,
        "stage_exhausted": updated_stage_exhausted,
        "no_new_sources_passes": next_no_new_sources_passes,
        "no_new_windows_passes": next_no_new_windows_passes,
        "no_new_visible_reels_passes": next_no_new_visible_reels_passes,
        "stage_is_exhausted": stage_is_exhausted,
        "all_stages_exhausted": next_stage > MAX_REFINEMENT_RECOVERY_STAGE,
        "growth_reason": growth_reason,
    }


def _build_refinement_request_params(
    *,
    creative_commons_only: bool,
    video_pool_mode: Literal["short-first", "balanced", "long-form"],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
    required_count: int,
    generation_mode: Literal["slow", "fast"],
    existing_source_count: int = 0,
    min_relevance_threshold: float = 0.0,
    page_hint: int = 1,
    page_size_hint: int = 5,
    target_page: int | None = None,
    recovery_stage: int = 0,
    refill_pass: int = 0,
    no_new_sources_passes: int = 0,
    no_new_windows_passes: int = 0,
    no_new_visible_reels_passes: int = 0,
    stage_exhausted: dict[str, bool] | None = None,
    growth_reason: str | None = None,
) -> dict[str, Any]:
    safe_required_count = max(1, int(required_count or 1))
    safe_page_hint = max(1, int(page_hint or 1))
    requested_target_page = max(safe_page_hint, int(target_page or safe_page_hint))
    safe_target_page = _refinement_horizon_page(
        required_count=safe_required_count,
        generation_mode=generation_mode,
        target_page=requested_target_page,
        page_size_hint=page_size_hint,
        existing_source_count=existing_source_count,
    )
    inventory_target_count = _refinement_inventory_target_count(
        required_count=safe_required_count,
        generation_mode=generation_mode,
        target_page=safe_target_page,
        page_size_hint=page_size_hint,
        existing_source_count=existing_source_count,
    )
    target_reel_count = _refinement_target_reel_count(
        required_count=safe_required_count,
        generation_mode=generation_mode,
        target_page=safe_target_page,
        page_size_hint=page_size_hint,
        existing_source_count=existing_source_count,
    )
    return {
        "creative_commons_only": creative_commons_only,
        "video_pool_mode": video_pool_mode,
        "preferred_video_duration": preferred_video_duration,
        "target_clip_duration_sec": target_clip_duration_sec,
        "target_clip_duration_min_sec": target_clip_duration_min_sec,
        "target_clip_duration_max_sec": target_clip_duration_max_sec,
        "required_reel_count": safe_required_count,
        "inventory_target_count": inventory_target_count,
        "target_reel_count": target_reel_count,
        "min_relevance_threshold": float(min_relevance_threshold or 0.0),
        "page_hint": safe_page_hint,
        "page_size_hint": max(1, int(page_size_hint or 1)),
        "target_page": safe_target_page,
        "recovery_stage": max(0, int(recovery_stage or 0)),
        "refill_pass": max(0, int(refill_pass or 0)),
        "no_new_sources_passes": max(0, int(no_new_sources_passes or 0)),
        "no_new_windows_passes": max(0, int(no_new_windows_passes or 0)),
        "no_new_visible_reels_passes": max(0, int(no_new_visible_reels_passes or 0)),
        "stage_exhausted": stage_exhausted or {},
        "growth_reason": str(growth_reason or "").strip(),
    }


def _merge_refinement_request_params(existing_job: dict[str, Any], request_params: dict[str, Any]) -> dict[str, Any]:
    try:
        existing_params = json.loads(str(existing_job.get("request_params_json") or "{}"))
    except json.JSONDecodeError:
        existing_params = {}
    if not isinstance(existing_params, dict):
        existing_params = {}
    merged = dict(existing_params)
    for key, value in request_params.items():
        if key in {"inventory_target_count", "target_reel_count"}:
            merged[key] = max(int(existing_params.get(key) or 0), int(value or 0))
        elif key == "required_reel_count":
            merged[key] = max(int(existing_params.get(key) or 0), int(value or 0))
        elif key in {
            "page_hint",
            "page_size_hint",
            "target_page",
            "recovery_stage",
            "refill_pass",
            "no_new_sources_passes",
            "no_new_windows_passes",
            "no_new_visible_reels_passes",
        }:
            merged[key] = max(int(existing_params.get(key) or 0), int(value or 0))
        elif key == "stage_exhausted":
            merged[key] = {
                **_normalize_stage_exhausted(existing_params.get(key)),
                **_normalize_stage_exhausted(value),
            }
        elif value is not None:
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    return merged


def _queue_refinement_if_needed(
    conn,
    *,
    material_id: str,
    concept_id: str | None,
    request_key: str,
    generation_id: str | None,
    current_reels: list[dict[str, Any]],
    required_count: int,
    generation_mode: Literal["slow", "fast"],
    creative_commons_only: bool,
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    video_pool_mode: Literal["short-first", "balanced", "long-form"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
    min_relevance: float | None,
    page_hint: int = 1,
    target_page: int | None = None,
    page_size_hint: int = 5,
    recovery_stage: int | None = None,
    refill_pass: int = 0,
    no_new_sources_passes: int = 0,
    no_new_windows_passes: int = 0,
    no_new_visible_reels_passes: int = 0,
    stage_exhausted: dict[str, bool] | None = None,
    growth_reason: str | None = None,
) -> dict[str, Any] | None:
    if not generation_id:
        return None
    safe_page_hint = max(1, int(page_hint or 1))
    safe_target_page = max(safe_page_hint, int(target_page or safe_page_hint))
    safe_page_size_hint = max(1, int(page_size_hint or 1))
    normalized_stage_exhausted = _normalize_stage_exhausted(stage_exhausted)
    inventory_target_count = _refinement_inventory_target_count(
        required_count=required_count,
        generation_mode=generation_mode,
        target_page=safe_target_page,
        page_size_hint=safe_page_size_hint,
        existing_source_count=len(current_reels),
    )
    target_reel_count = _refinement_target_reel_count(
        required_count=required_count,
        generation_mode=generation_mode,
        target_page=safe_target_page,
        page_size_hint=safe_page_size_hint,
        existing_source_count=len(current_reels),
    )
    if len(current_reels) >= inventory_target_count or target_reel_count <= 0:
        return _fetch_refinement_job_for_generation(conn, generation_id)
    if recovery_stage is None:
        safe_recovery_stage = (
            _first_unexhausted_recovery_stage(normalized_stage_exhausted)
            if normalized_stage_exhausted
            else _initial_recovery_stage(page_hint=safe_page_hint)
        )
    else:
        safe_recovery_stage = max(0, int(recovery_stage or 0))
    if safe_recovery_stage > MAX_REFINEMENT_RECOVERY_STAGE:
        return _fetch_refinement_job_for_generation(conn, generation_id)
    return _queue_refinement_job(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        request_key=request_key,
        source_generation_id=generation_id,
        request_params=_build_refinement_request_params(
            creative_commons_only=creative_commons_only,
            video_pool_mode=video_pool_mode,
            preferred_video_duration=preferred_video_duration,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
            required_count=required_count,
            generation_mode=generation_mode,
            existing_source_count=len(current_reels),
            min_relevance_threshold=float(min_relevance or 0.0),
            page_hint=safe_page_hint,
            page_size_hint=safe_page_size_hint,
            target_page=safe_target_page,
            recovery_stage=safe_recovery_stage,
            refill_pass=refill_pass,
            no_new_sources_passes=no_new_sources_passes,
            no_new_windows_passes=no_new_windows_passes,
            no_new_visible_reels_passes=no_new_visible_reels_passes,
            stage_exhausted=normalized_stage_exhausted,
            growth_reason=growth_reason,
        ),
    )


def _queue_refinement_job(
    conn,
    *,
    material_id: str,
    concept_id: str | None,
    request_key: str,
    source_generation_id: str,
    request_params: dict[str, Any],
) -> dict[str, Any] | None:
    if SERVERLESS_MODE or _refinement_job_executor is None:
        return None

    existing = _fetch_refinement_job_for_generation(conn, source_generation_id)
    if existing:
        merged_params = _merge_refinement_request_params(existing, request_params)
        if dumps_json(merged_params) != str(existing.get("request_params_json") or "{}"):
            updated_existing = dict(existing)
            updated_existing["request_params_json"] = dumps_json(merged_params)
            upsert(conn, "reel_generation_jobs", updated_existing)
            return updated_existing
        return existing

    job_id = str(uuid.uuid4())
    created_at = now_iso()
    job_row = {
        "id": job_id,
        "material_id": material_id,
        "concept_id": concept_id,
        "request_key": request_key,
        "source_generation_id": source_generation_id,
        "result_generation_id": None,
        "target_profile": "deep",
        "request_params_json": dumps_json(request_params),
        "status": "queued",
        "created_at": created_at,
        "started_at": None,
        "completed_at": None,
        "error_text": None,
    }
    upsert(conn, "reel_generation_jobs", job_row)

    _submit_refinement_job(job_id)
    return job_row


def _run_refinement_job(job_id: str) -> None:
    with get_conn() as conn:
        job_row = fetch_one(conn, "SELECT * FROM reel_generation_jobs WHERE id = ?", (job_id,))
        if not job_row:
            return
        if str(job_row.get("status") or "").strip().lower() != "queued":
            return

        started_at = now_iso()
        claimed = execute_modify(
            conn,
            """
            UPDATE reel_generation_jobs
            SET status = 'running',
                started_at = ?,
                completed_at = NULL,
                error_text = NULL
            WHERE id = ?
              AND status = 'queued'
            """,
            (started_at, job_id),
        )
        if claimed <= 0:
            return
        job_row = fetch_one(conn, "SELECT * FROM reel_generation_jobs WHERE id = ?", (job_id,))
        if not job_row:
            return

        source_generation_id = str(job_row.get("source_generation_id") or "")
        source_generation = _fetch_generation_row(conn, source_generation_id)
        if not source_generation:
            upsert(
                conn,
                "reel_generation_jobs",
                {
                    **job_row,
                    "status": "failed",
                    "completed_at": now_iso(),
                    "error_text": "source_generation_missing",
                },
            )
            return

        result_generation_id = _create_generation_row(
            conn,
            material_id=str(job_row.get("material_id") or ""),
            concept_id=str(job_row.get("concept_id") or "") or None,
            request_key=str(job_row.get("request_key") or ""),
            generation_mode=str(source_generation.get("generation_mode") or "fast"),  # type: ignore[arg-type]
            retrieval_profile="deep",
            source_generation_id=source_generation_id,
        )

        try:
            try:
                request_params = json.loads(str(job_row.get("request_params_json") or "{}"))
            except json.JSONDecodeError:
                request_params = {}
            if not isinstance(request_params, dict):
                request_params = {}
            fast_mode = str(source_generation.get("generation_mode") or "fast") == "fast"
            generation_mode: Literal["slow", "fast"] = "fast" if fast_mode else "slow"
            safe_video_pool_mode = _normalize_video_pool_mode(str(request_params.get("video_pool_mode") or "short-first"))
            safe_video_duration_pref = _normalize_preferred_video_duration(
                str(request_params.get("preferred_video_duration") or "any")
            )
            safe_target_clip_duration_sec = int(request_params.get("target_clip_duration_sec") or DEFAULT_TARGET_CLIP_DURATION_SEC)
            safe_target_clip_min_sec = (
                int(request_params["target_clip_duration_min_sec"])
                if request_params.get("target_clip_duration_min_sec") is not None
                else MIN_TARGET_CLIP_DURATION_SEC
            )
            safe_target_clip_max_sec = (
                int(request_params["target_clip_duration_max_sec"])
                if request_params.get("target_clip_duration_max_sec") is not None
                else MAX_TARGET_CLIP_DURATION_SEC
            )
            safe_min_relevance = float(request_params.get("min_relevance_threshold") or 0.0)
            safe_page_hint = max(1, int(request_params.get("page_hint") or 1))
            safe_page_size_hint = max(1, int(request_params.get("page_size_hint") or 5))
            safe_target_page = max(safe_page_hint, int(request_params.get("target_page") or safe_page_hint))
            normalized_stage_exhausted = _normalize_stage_exhausted(request_params.get("stage_exhausted"))
            safe_recovery_stage = max(
                0,
                int(
                    request_params.get("recovery_stage")
                    if request_params.get("recovery_stage") is not None
                    else _first_unexhausted_recovery_stage(normalized_stage_exhausted)
                ),
            )
            safe_refill_pass = max(0, int(request_params.get("refill_pass") or 0))
            safe_no_new_sources_passes = max(0, int(request_params.get("no_new_sources_passes") or 0))
            safe_no_new_windows_passes = max(0, int(request_params.get("no_new_windows_passes") or 0))
            safe_no_new_visible_reels_passes = max(0, int(request_params.get("no_new_visible_reels_passes") or 0))
            safe_required_reel_count = max(1, int(request_params.get("required_reel_count") or request_params.get("target_reel_count") or 1))
            source_reel_count = int(source_generation.get("reel_count") or 0)
            inventory_target_count = max(
                safe_required_reel_count,
                int(request_params.get("inventory_target_count") or 0),
                _refinement_inventory_target_count(
                    required_count=safe_required_reel_count,
                    generation_mode=generation_mode,
                    target_page=safe_target_page,
                    page_size_hint=safe_page_size_hint,
                    existing_source_count=source_reel_count,
                ),
            )
            target_reel_count = _refinement_target_reel_count(
                required_count=safe_required_reel_count,
                generation_mode=generation_mode,
                target_page=safe_target_page,
                page_size_hint=safe_page_size_hint,
                existing_source_count=source_reel_count,
            )
            source_visible_reels = _ranked_request_reels(
                conn,
                material_id=str(job_row.get("material_id") or ""),
                fast_mode=fast_mode,
                generation_id=source_generation_id,
                min_relevance=safe_min_relevance,
                preferred_video_duration=safe_video_duration_pref,
                target_clip_duration_sec=safe_target_clip_duration_sec,
                target_clip_duration_min_sec=safe_target_clip_min_sec,
                target_clip_duration_max_sec=safe_target_clip_max_sec,
                page=safe_target_page,
                limit=safe_page_size_hint,
            )
            source_visible_keys = _visible_reel_clip_keys(source_visible_reels)
            _publish_progress_event(
                "stage_start",
                stage=f"refinement_stage_{safe_recovery_stage}",
                recovery_stage=safe_recovery_stage,
                target_reel_count=target_reel_count,
                generation_id=result_generation_id,
                generation_mode=generation_mode,
            )
            stage_started_mono = time.monotonic()
            reel_service.generate_reels(
                conn,
                material_id=str(job_row.get("material_id") or ""),
                concept_id=str(job_row.get("concept_id") or "") or None,
                num_reels=target_reel_count,
                creative_commons_only=bool(request_params.get("creative_commons_only", False)),
                fast_mode=fast_mode,
                video_pool_mode=safe_video_pool_mode,
                preferred_video_duration=safe_video_duration_pref,
                target_clip_duration_sec=safe_target_clip_duration_sec,
                target_clip_duration_min_sec=safe_target_clip_min_sec,
                target_clip_duration_max_sec=safe_target_clip_max_sec,
                retrieval_profile="deep",
                generation_id=result_generation_id,
                exclude_generation_ids=_response_generation_ids(conn, source_generation_id) or [source_generation_id],
                min_relevance_threshold=safe_min_relevance,
                page_hint=safe_target_page,
                recovery_stage=safe_recovery_stage,
            )
            _publish_progress_event(
                "stage_end",
                stage=f"refinement_stage_{safe_recovery_stage}",
                recovery_stage=safe_recovery_stage,
                duration_ms=int((time.monotonic() - stage_started_mono) * 1000),
                outcome="ok",
            )
        except Exception as exc:
            _publish_progress_event(
                "stage_end",
                stage=f"refinement_stage_{safe_recovery_stage}",
                recovery_stage=safe_recovery_stage,
                outcome="error",
                error_class=type(exc).__name__,
                detail=str(exc)[:240],
            )
            _complete_generation(
                conn,
                generation_id=result_generation_id,
                retrieval_profile="deep",
                status="failed",
                error_text=str(exc),
            )
            upsert(
                conn,
                "reel_generation_jobs",
                {
                    **job_row,
                    "result_generation_id": result_generation_id,
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": now_iso(),
                    "error_text": str(exc),
                },
            )
            return

        result_count = _count_generation_reels(conn, result_generation_id)
        current_active = _fetch_active_generation_row(
            conn,
            material_id=str(job_row.get("material_id") or ""),
            request_key=str(job_row.get("request_key") or ""),
        )
        current_active_id = str((current_active or {}).get("id") or "")
        next_state = _advance_refinement_state(
            recovery_stage=safe_recovery_stage,
            stage_exhausted=normalized_stage_exhausted,
            no_new_sources_passes=safe_no_new_sources_passes,
            no_new_windows_passes=safe_no_new_windows_passes,
            no_new_visible_reels_passes=safe_no_new_visible_reels_passes,
            new_unique_videos=_count_generation_unique_videos(conn, result_generation_id) if result_count > 0 else 0,
            new_unique_clip_windows=result_count,
            new_unique_visible_reels=0,
        )

        if result_count <= 0:
            _complete_generation(
                conn,
                generation_id=result_generation_id,
                retrieval_profile="deep",
                status="failed",
                error_text="all_stages_exhausted" if next_state["all_stages_exhausted"] else "no_reels_generated",
            )
            upsert(
                conn,
                "reel_generation_jobs",
                {
                    **job_row,
                    "result_generation_id": result_generation_id,
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": now_iso(),
                    "error_text": "all_stages_exhausted" if next_state["all_stages_exhausted"] else "no_reels_generated",
                },
            )
            return

        if current_active_id != source_generation_id:
            _complete_generation(
                conn,
                generation_id=result_generation_id,
                retrieval_profile="deep",
                status="completed",
            )
            upsert(
                conn,
                "reel_generation_jobs",
                {
                    **job_row,
                    "result_generation_id": result_generation_id,
                    "status": "superseded",
                    "started_at": started_at,
                    "completed_at": now_iso(),
                    "error_text": None,
                },
            )
            return

        _activate_generation(
            conn,
            material_id=str(job_row.get("material_id") or ""),
            request_key=str(job_row.get("request_key") or ""),
            generation_id=result_generation_id,
            retrieval_profile="deep",
        )
        upsert(
            conn,
            "reel_generation_jobs",
            {
                **job_row,
                "result_generation_id": result_generation_id,
                "status": "completed",
                "started_at": started_at,
                "completed_at": now_iso(),
                "error_text": None,
            },
        )
        merged_reels = _ranked_request_reels(
            conn,
            material_id=str(job_row.get("material_id") or ""),
            fast_mode=fast_mode,
            generation_id=result_generation_id,
            min_relevance=safe_min_relevance,
            preferred_video_duration=safe_video_duration_pref,
            target_clip_duration_sec=safe_target_clip_duration_sec,
            target_clip_duration_min_sec=safe_target_clip_min_sec,
            target_clip_duration_max_sec=safe_target_clip_max_sec,
            page=safe_target_page,
            limit=safe_page_size_hint,
        )
        merged_visible_keys = _visible_reel_clip_keys(merged_reels)
        new_visible_reels: list[dict[str, Any]] = []
        for reel in merged_reels:
            _reel_id, clip_key = _reel_identity_key(reel)
            if clip_key and clip_key not in source_visible_keys:
                new_visible_reels.append(reel)
        _update_frontier_visible_growth(
            conn,
            material_id=str(job_row.get("material_id") or ""),
            request_key=str(job_row.get("request_key") or ""),
            reels=new_visible_reels,
        )
        next_state = _advance_refinement_state(
            recovery_stage=safe_recovery_stage,
            stage_exhausted=normalized_stage_exhausted,
            no_new_sources_passes=safe_no_new_sources_passes,
            no_new_windows_passes=safe_no_new_windows_passes,
            no_new_visible_reels_passes=safe_no_new_visible_reels_passes,
            new_unique_videos=_count_generation_unique_videos(conn, result_generation_id),
            new_unique_clip_windows=result_count,
            new_unique_visible_reels=len(new_visible_reels),
        )
        _publish_progress_event(
            "reels_completed",
            result_generation_id=result_generation_id,
            total_clip_windows=result_count,
            new_visible_reels=len(new_visible_reels),
            recovery_stage=safe_recovery_stage,
            all_stages_exhausted=bool(next_state.get("all_stages_exhausted")),
        )
        # Keep refinement client-driven. A single queued job may finish after the
        # user leaves the page, but it should not recursively schedule more work
        # on its own. If the client is still active and needs more inventory, the
        # next /api/feed or /api/reels/generate request will enqueue another pass.


def _ensure_generation_for_request(
    conn,
    *,
    material_id: str,
    concept_id: str | None,
    required_count: int,
    sync_deep_fallback: Literal["always", "if_empty", "never"] = "always",
    creative_commons_only: bool,
    generation_mode: Literal["slow", "fast"],
    min_relevance: float | None,
    video_pool_mode: Literal["short-first", "balanced", "long-form"],
    preferred_video_duration: Literal["any", "short", "medium", "long"],
    target_clip_duration_sec: int,
    target_clip_duration_min_sec: int | None,
    target_clip_duration_max_sec: int | None,
    exclude_video_ids: list[str] | None = None,
    page_hint: int = 1,
    page_size_hint: int = 5,
    on_reel_created: Callable[[dict[str, Any]], None] | None = None,
    emit_existing_reels: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if should_cancel is not None and should_cancel():
        raise GenerationCancelledError("Generation cancelled.")
    request_key = _build_generation_request_key(
        material_id=material_id,
        concept_id=concept_id,
        creative_commons_only=creative_commons_only,
        generation_mode=generation_mode,
        video_pool_mode=video_pool_mode,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
    )
    fast_mode = generation_mode == "fast"
    normalized_excluded_video_ids = _normalize_excluded_video_ids(exclude_video_ids)
    bootstrap_target = _initial_response_reel_target(
        required_count=required_count,
        generation_mode=generation_mode,
        sync_deep_fallback=sync_deep_fallback,
    )
    minimum_initial_response_reels = _minimum_initial_response_reels(
        required_count=required_count,
        generation_mode=generation_mode,
        sync_deep_fallback=sync_deep_fallback,
    )
    active_generation = _fetch_active_generation_row(conn, material_id=material_id, request_key=request_key)
    if active_generation:
        active_generation_id = str(active_generation.get("id") or "")
        active_response_profile = str(active_generation.get("retrieval_profile") or "") or None
        active_reels = _ranked_request_reels(
            conn,
            material_id=material_id,
            fast_mode=fast_mode,
            generation_id=active_generation_id,
            min_relevance=min_relevance,
            preferred_video_duration=preferred_video_duration,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
            exclude_video_ids=normalized_excluded_video_ids,
            page=page_hint,
            limit=max(page_size_hint, min(25, required_count)),
        )
        active_job = _fetch_refinement_job_for_generation(conn, active_generation_id)
        active_job = _queue_refinement_if_needed(
            conn,
            material_id=material_id,
            concept_id=concept_id,
            request_key=request_key,
            generation_id=active_generation_id,
            current_reels=active_reels,
            required_count=required_count,
            generation_mode=generation_mode,
            creative_commons_only=creative_commons_only,
            preferred_video_duration=preferred_video_duration,
            video_pool_mode=video_pool_mode,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
            min_relevance=min_relevance,
            page_hint=page_hint,
            page_size_hint=max(1, int(page_size_hint or 1)),
        )
        if emit_existing_reels and on_reel_created is not None:
            for reel in active_reels[: max(1, required_count)]:
                if should_cancel is not None and should_cancel():
                    raise GenerationCancelledError("Generation cancelled.")
                try:
                    on_reel_created(reel)
                except Exception:
                    pass
        if len(active_reels) < required_count:
            should_extend_active_generation = active_response_profile != "bootstrap" or (
                sync_deep_fallback != "never" and len(active_reels) < minimum_initial_response_reels
            )
            if should_extend_active_generation:
                return _extend_active_generation(
                    conn,
                    material_id=material_id,
                    concept_id=concept_id,
                    request_key=request_key,
                    required_count=required_count,
                    active_generation_id=active_generation_id,
                    active_response_profile=active_response_profile,
                    generation_mode=generation_mode,
                    creative_commons_only=creative_commons_only,
                    min_relevance=min_relevance,
                    video_pool_mode=video_pool_mode,
                    preferred_video_duration=preferred_video_duration,
                    target_clip_duration_sec=target_clip_duration_sec,
                    target_clip_duration_min_sec=target_clip_duration_min_sec,
                    target_clip_duration_max_sec=target_clip_duration_max_sec,
                    exclude_video_ids=normalized_excluded_video_ids,
                    page_hint=page_hint,
                    on_reel_created=on_reel_created,
                    should_cancel=should_cancel,
                )
        if (
            len(active_reels) >= required_count
            or (
                active_reels
                and sync_deep_fallback != "always"
                and active_response_profile == "bootstrap"
                and len(active_reels) >= minimum_initial_response_reels
            )
        ):
            response_payload = _build_generation_response_payload(
                reels=active_reels,
                generation_id=active_generation_id,
                response_profile=active_response_profile,
                job_row=active_job,
            )
            _log_generation_response_summary(
                event="reuse_active_generation",
                material_id=material_id,
                request_key=request_key,
                generation_mode=generation_mode,
                required_count=required_count,
                response_payload=response_payload,
                page_hint=page_hint,
            )
            return {**response_payload, "request_key": request_key}

    generation_id = _create_generation_row(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        request_key=request_key,
        generation_mode=generation_mode,
        retrieval_profile="bootstrap",
        source_generation_id=str((active_generation or {}).get("id") or "") or None,
    )

    reel_service.generate_reels(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        num_reels=bootstrap_target,
        creative_commons_only=creative_commons_only,
        exclude_video_ids=normalized_excluded_video_ids,
        fast_mode=fast_mode,
        video_pool_mode=video_pool_mode,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        retrieval_profile="bootstrap",
        generation_id=generation_id,
        min_relevance_threshold=float(min_relevance or 0.0),
        page_hint=page_hint,
        recovery_stage=0,
        on_reel_created=on_reel_created,
        should_cancel=should_cancel,
    )
    filtered = _ranked_request_reels(
        conn,
        material_id=material_id,
        fast_mode=fast_mode,
        generation_id=generation_id,
        min_relevance=min_relevance,
        preferred_video_duration=preferred_video_duration,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        exclude_video_ids=normalized_excluded_video_ids,
        page=page_hint,
        limit=max(page_size_hint, min(25, required_count)),
    )

    response_profile = "bootstrap"
    should_run_sync_deep = False
    if not filtered:
        should_run_sync_deep = sync_deep_fallback in {"always", "if_empty"}
    elif len(filtered) < minimum_initial_response_reels:
        should_run_sync_deep = sync_deep_fallback in {"always", "if_empty"}
    elif len(filtered) < required_count:
        should_run_sync_deep = sync_deep_fallback == "always"
    if should_run_sync_deep:
        shortfall = max(1, required_count - len(filtered))
        reel_service.generate_reels(
            conn,
            material_id=material_id,
            concept_id=concept_id,
            num_reels=shortfall,
            creative_commons_only=creative_commons_only,
            exclude_video_ids=normalized_excluded_video_ids,
            fast_mode=fast_mode,
            video_pool_mode=video_pool_mode,
            preferred_video_duration=preferred_video_duration,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
            retrieval_profile="deep",
            generation_id=generation_id,
            min_relevance_threshold=float(min_relevance or 0.0),
            page_hint=page_hint,
            recovery_stage=_initial_recovery_stage(page_hint=page_hint),
            on_reel_created=on_reel_created,
            should_cancel=should_cancel,
        )
        filtered = _ranked_request_reels(
            conn,
            material_id=material_id,
            fast_mode=fast_mode,
            generation_id=generation_id,
            min_relevance=min_relevance,
            preferred_video_duration=preferred_video_duration,
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
            exclude_video_ids=normalized_excluded_video_ids,
        )
        if filtered:
            response_profile = "bootstrap_then_deep"

    if not filtered:
        _complete_generation(
            conn,
            generation_id=generation_id,
            retrieval_profile=response_profile,
            status="failed",
            error_text="no_reels_generated",
        )
        fallback_reels = []
        if active_generation:
            fallback_reels = _ranked_request_reels(
                conn,
                material_id=material_id,
                fast_mode=fast_mode,
                generation_id=str(active_generation.get("id") or ""),
                min_relevance=min_relevance,
                preferred_video_duration=preferred_video_duration,
                target_clip_duration_sec=target_clip_duration_sec,
                target_clip_duration_min_sec=target_clip_duration_min_sec,
                target_clip_duration_max_sec=target_clip_duration_max_sec,
                exclude_video_ids=normalized_excluded_video_ids,
            )
        active_job = _fetch_refinement_job_for_generation(conn, str((active_generation or {}).get("id") or ""))
        response_payload = _build_generation_response_payload(
            reels=fallback_reels,
            generation_id=str((active_generation or {}).get("id") or "") or None,
            response_profile=str((active_generation or {}).get("retrieval_profile") or "") or None,
            job_row=active_job,
        )
        _log_generation_response_summary(
            event="new_generation_empty",
            material_id=material_id,
            request_key=request_key,
            generation_mode=generation_mode,
            required_count=required_count,
            response_payload=response_payload,
            page_hint=page_hint,
        )
        return {**response_payload, "request_key": request_key}

    _activate_generation(
        conn,
        material_id=material_id,
        request_key=request_key,
        generation_id=generation_id,
        retrieval_profile=response_profile,
    )
    job_row = _queue_refinement_if_needed(
        conn,
        material_id=material_id,
        concept_id=concept_id,
        request_key=request_key,
        generation_id=generation_id,
        current_reels=filtered,
        required_count=required_count,
        generation_mode=generation_mode,
        creative_commons_only=creative_commons_only,
        preferred_video_duration=preferred_video_duration,
        video_pool_mode=video_pool_mode,
        target_clip_duration_sec=target_clip_duration_sec,
        target_clip_duration_min_sec=target_clip_duration_min_sec,
        target_clip_duration_max_sec=target_clip_duration_max_sec,
        min_relevance=min_relevance,
        page_hint=1,
        page_size_hint=max(1, int(page_size_hint or 1)),
    )
    response_payload = _build_generation_response_payload(
        reels=filtered,
        generation_id=generation_id,
        response_profile=response_profile,
        job_row=job_row,
    )
    _log_generation_response_summary(
        event="new_generation_complete",
        material_id=material_id,
        request_key=request_key,
        generation_mode=generation_mode,
        required_count=required_count,
        response_payload=response_payload,
        page_hint=page_hint,
    )
    return {**response_payload, "request_key": request_key}


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


def _serialize_community_history_item(row: dict) -> CommunityHistoryItemOut:
    generation_mode = str(row.get("generation_mode") or "").strip().lower()
    if generation_mode not in {"slow", "fast"}:
        generation_mode = "fast"
    source = str(row.get("source") or "").strip().lower()
    if source not in {"search", "community"}:
        source = "search"
    feed_query_raw = str(row.get("feed_query") or "").strip()
    active_index = _to_int(row.get("active_index"), -1)
    active_reel_id_raw = str(row.get("active_reel_id") or "").strip()
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
            generation_mode = "fast"
        source = str(item.source or "").strip().lower()
        if source not in {"search", "community"}:
            source = "search"
        feed_query_raw = str(item.feed_query or "").strip()
        active_index = max(0, int(item.active_index)) if item.active_index is not None else None
        active_reel_id_raw = str(item.active_reel_id or "").strip()
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
):
    _enforce_rate_limit(request, "material", limit=MATERIAL_RATE_LIMIT_PER_WINDOW)
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
def generate_reels(request: Request, payload: ReelsGenerateRequest):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "reels-generate", limit=REELS_GENERATE_RATE_LIMIT_PER_WINDOW)
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
    with get_conn() as conn:
        material = fetch_all(conn, "SELECT id FROM materials WHERE id = ?", (payload.material_id,))
        if not material:
            # Keep the string detail form so existing deployed clients that
            # substring-match on "material_id not found" continue to work.
            # iOS/web clients should prefer HTTP 404 + this marker over
            # localised messages when routing to recovery.
            raise HTTPException(status_code=404, detail="material_id not found")

        effective_generation_mode: Literal["slow", "fast"] = "fast" if SERVERLESS_MODE else payload.generation_mode
        try:
            generation_result = _ensure_generation_for_request(
                conn,
                material_id=payload.material_id,
                concept_id=payload.concept_id,
                required_count=requested_num_reels,
                sync_deep_fallback="if_empty",
                creative_commons_only=payload.creative_commons_only,
                generation_mode=effective_generation_mode,
                min_relevance=min_relevance,
                video_pool_mode=safe_video_pool_mode,
                preferred_video_duration=safe_video_duration_pref,
                target_clip_duration_sec=safe_target_clip_duration_sec,
                target_clip_duration_min_sec=safe_target_clip_min_sec,
                target_clip_duration_max_sec=safe_target_clip_max_sec,
                exclude_video_ids=payload.exclude_video_ids,
                page_hint=1,
            )
        except YouTubeApiRequestError as e:
            logger.warning(
                "Reels generate request failed: %s",
                json.dumps(
                    {
                        "material_id": payload.material_id,
                        "generation_mode": effective_generation_mode,
                        "requested_num_reels": requested_num_reels,
                        "error": str(e),
                    },
                    sort_keys=True,
                ),
            )
            raise HTTPException(status_code=502, detail=str(e)) from e
    return {
        "reels": list(generation_result.get("reels") or [])[:requested_num_reels],
        "generation_id": generation_result.get("generation_id"),
        "response_profile": generation_result.get("response_profile"),
        "refinement_job_id": generation_result.get("refinement_job_id"),
        "refinement_status": generation_result.get("refinement_status"),
    }


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


@app.post("/api/ingest/url", response_model=IngestResult)
def ingest_url_endpoint(request: Request, payload: IngestRequest) -> IngestResult:
    """
    Download + process a single reel URL (YouTube / Instagram / TikTok) and persist a
    ReelOut-compatible record. See `app/ingestion/__init__.py` for the full design notes
    and legal posture.
    """
    _enforce_rate_limit(request, "ingest-url", limit=INGEST_URL_RATE_LIMIT_PER_WINDOW)
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Reel ingestion is disabled in serverless mode.",
            },
        )

    try:
        result = ingestion_pipeline.ingest_url(
            source_url=payload.source_url,
            material_id=payload.material_id,
            concept_id=payload.concept_id,
            target_clip_duration_sec=payload.target_clip_duration_sec,
            target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
            target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
            language=payload.language,
        )
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
def ingest_topic_cut_endpoint(request: Request, payload: IngestTopicCutRequest) -> IngestTopicCutResult:
    """
    Topic-aware variant of `/api/ingest/url`. Same download + transcribe path,
    but instead of returning ONE clip per video this endpoint returns one reel
    per topic the creator introduces (Shorts are returned with `reels: []` and
    `is_short: true` so the caller can leave them untouched).

    Each entry in `reels` decodes cleanly into the iOS `Reel` struct via the
    existing decoder — no client schema changes are required.
    """
    _enforce_rate_limit(request, "ingest-topic-cut", limit=INGEST_TOPIC_CUT_RATE_LIMIT_PER_WINDOW)
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Topic-cut ingestion is disabled in serverless mode.",
            },
        )

    try:
        result = ingestion_pipeline.ingest_topic_cut(
            source_url=payload.source_url,
            material_id=payload.material_id,
            concept_id=payload.concept_id,
            language=payload.language,
            use_llm=payload.use_llm,
            query=payload.query,
        )
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
def ingest_search_endpoint(request: Request, payload: IngestSearchRequest) -> IngestSearchResult:
    """
    Topic-based multi-platform search. Resolves YouTube + Instagram + TikTok search
    results via yt-dlp's native extractors, dedups against `exclude_video_ids` for
    infinite-scroll pagination, and ingests each resolved URL through the full pipeline
    (Whisper fallback, silence-aware cuts, full metadata). Per-platform failures are
    non-fatal — the response carries `per_platform_errors` so the client can surface
    which platform went down.
    """
    _enforce_rate_limit(request, "ingest-search", limit=INGEST_SEARCH_RATE_LIMIT_PER_WINDOW)
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Reel ingestion is disabled in serverless mode.",
            },
        )

    try:
        result = ingestion_pipeline.ingest_search(
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
        )
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
def ingest_feed_endpoint(request: Request, payload: IngestFeedRequest) -> IngestFeedResult:
    """
    Resolve a profile / hashtag / playlist URL to a bounded list of individual reel URLs
    and ingest each in a small thread pool. Per-item failures are recorded in the response
    (`items[*].status == "error"`) rather than aborting the whole call.
    """
    _enforce_rate_limit(request, "ingest-feed", limit=INGEST_FEED_RATE_LIMIT_PER_WINDOW)
    if SERVERLESS_MODE and os.getenv("ALLOW_OPENAI_IN_SERVERLESS") != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ServerlessUnavailable",
                "message": "Reel ingestion is disabled in serverless mode.",
            },
        )

    try:
        result = ingestion_pipeline.ingest_feed(
            feed_url=payload.feed_url,
            max_items=payload.max_items,
            material_id=payload.material_id,
            concept_id=payload.concept_id,
            target_clip_duration_sec=payload.target_clip_duration_sec,
            target_clip_duration_min_sec=payload.target_clip_duration_min_sec,
            target_clip_duration_max_sec=payload.target_clip_duration_max_sec,
            language=payload.language,
        )
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


@app.post("/api/reels/generate-stream")
async def generate_reels_stream(request: Request, payload: ReelsGenerateRequest):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "reels-generate", limit=REELS_GENERATE_RATE_LIMIT_PER_WINDOW)
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
        requested_num_reels = min(requested_num_reels, 6)

    with get_conn() as conn:
        material = fetch_all(conn, "SELECT id FROM materials WHERE id = ?", (payload.material_id,))
        if not material:
            # Keep the string detail form so existing deployed clients that
            # substring-match on "material_id not found" continue to work.
            # iOS/web clients should prefer HTTP 404 + this marker over
            # localised messages when routing to recovery.
            raise HTTPException(status_code=404, detail="material_id not found")

    effective_generation_mode: Literal["slow", "fast"] = "fast" if SERVERLESS_MODE else payload.generation_mode

    async def event_stream():
        # Bounded queue so a slow client back-pressures into the worker instead of
        # letting it allocate unbounded memory ahead of the network drain.
        event_queue: Queue[dict[str, Any] | None] = Queue(maxsize=256)
        emitted_reels: set[tuple[str, str]] = set()
        cancel_event = threading.Event()
        # Hard wall-clock deadline for the whole generation so a runaway LLM /
        # YouTube call cannot bill LLM credits forever for a disconnected client.
        STREAM_DEADLINE_SEC = 25.0 if SERVERLESS_MODE else 90.0
        deadline = time.monotonic() + STREAM_DEADLINE_SEC

        def emit_event(event: dict[str, Any]) -> None:
            try:
                event_queue.put(event, timeout=STREAM_DEADLINE_SEC)
            except Exception:
                # If the queue is still full after the deadline the client has
                # almost certainly gone away. Signal cancel and drop the event
                # rather than blocking the worker thread indefinitely.
                cancel_event.set()

        def emit_reel(reel: dict[str, Any]) -> None:
            reel_id, clip_key = _reel_identity_key(reel)
            identity = (reel_id, clip_key)
            if identity in emitted_reels:
                return
            emitted_reels.add(identity)
            emit_event({"type": "reel", "reel": reel})

        def should_cancel_or_deadline() -> bool:
            if cancel_event.is_set():
                return True
            if time.monotonic() >= deadline:
                cancel_event.set()
                return True
            return False

        def run_generation() -> None:
            try:
                with get_conn() as conn:
                    generation_result = _ensure_generation_for_request(
                        conn,
                        material_id=payload.material_id,
                        concept_id=payload.concept_id,
                        required_count=requested_num_reels,
                        sync_deep_fallback="if_empty",
                        creative_commons_only=payload.creative_commons_only,
                        generation_mode=effective_generation_mode,
                        min_relevance=min_relevance,
                        video_pool_mode=safe_video_pool_mode,
                        preferred_video_duration=safe_video_duration_pref,
                        target_clip_duration_sec=safe_target_clip_duration_sec,
                        target_clip_duration_min_sec=safe_target_clip_min_sec,
                        target_clip_duration_max_sec=safe_target_clip_max_sec,
                        exclude_video_ids=payload.exclude_video_ids,
                        page_hint=1,
                        on_reel_created=emit_reel,
                        emit_existing_reels=True,
                        should_cancel=should_cancel_or_deadline,
                    )
                emit_event(
                    {
                        "type": "done",
                        "response": {
                            "reels": list(generation_result.get("reels") or [])[:requested_num_reels],
                            "generation_id": generation_result.get("generation_id"),
                            "response_profile": generation_result.get("response_profile"),
                            "refinement_job_id": generation_result.get("refinement_job_id"),
                            "refinement_status": generation_result.get("refinement_status"),
                        },
                    }
                )
            except GenerationCancelledError:
                pass
            except YouTubeApiRequestError as exc:
                logger.warning(
                    "Reels generate-stream request failed: %s",
                    json.dumps(
                        {
                            "material_id": payload.material_id,
                            "generation_mode": effective_generation_mode,
                            "requested_num_reels": requested_num_reels,
                            "error": str(exc),
                        },
                        sort_keys=True,
                    ),
                )
                emit_event({"type": "error", "detail": str(exc)})
            except HTTPException as exc:
                emit_event({"type": "error", "detail": str(exc.detail), "status_code": exc.status_code})
            except Exception as exc:
                logger.exception(
                    "Reels generate-stream unexpected failure material_id=%s mode=%s requested=%s",
                    payload.material_id,
                    effective_generation_mode,
                    requested_num_reels,
                )
                emit_event({"type": "error", "detail": str(exc)})
            finally:
                event_queue.put(None)

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()

        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                try:
                    event = event_queue.get(timeout=0.25)
                except Empty:
                    if cancel_event.is_set() or (not worker.is_alive() and event_queue.empty()):
                        break
                    continue
                if event is None:
                    break
                yield f"{json.dumps(event)}\n"
        finally:
            cancel_event.set()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/api/reels/refinement-stream/{job_id}")
async def refinement_stream(request: Request, job_id: str, replay_from_ms: int = 0):
    """
    NDJSON progress stream for a refinement job (Phase D.1).

    Emits one JSON line per progress event (stage_start, stage_end, reels_completed,
    job_start, job_end, job_terminal, and any custom events published by the worker).
    The response continues until the worker emits a `job_terminal` event (clean EOF)
    or the subscriber's idle timeout elapses.

    Late subscribers can pass `?replay_from_ms=` to skip events they already saw on
    reconnect. When `replay_from_ms=0` (default) the client gets the full buffered
    history (up to `progress_bus.BUFFER_SIZE` events).

    Both iOS (APIClient.swift:589) and the webapp (api.ts:925) already handle
    line-buffered NDJSON for /api/reels/generate-stream — this endpoint reuses that
    client-side plumbing.
    """
    _require_community_client_identity(request)
    _enforce_rate_limit(
        request,
        "reels-refinement-stream",
        limit=REELS_REFINEMENT_STATUS_RATE_LIMIT_PER_WINDOW,
    )

    clean_job_id = str(job_id or "").strip()
    if not clean_job_id:
        raise HTTPException(status_code=400, detail="job_id required")

    with get_conn() as conn:
        job_row = fetch_one(
            conn, "SELECT id FROM reel_generation_jobs WHERE id = ? LIMIT 1", (clean_job_id,)
        )
    if not job_row:
        raise HTTPException(status_code=404, detail="job_id not found")

    async def ndjson_events():
        try:
            async for event in progress_bus.subscribe(
                clean_job_id,
                replay_from_ms=int(replay_from_ms or 0),
                idle_timeout_sec=90.0,
            ):
                if await request.is_disconnected():
                    break
                yield f"{json.dumps(event, default=str)}\n"
        except Exception as exc:  # pragma: no cover - defensive
            yield f"{json.dumps({'event': 'stream_error', 'detail': str(exc)})}\n"

    return StreamingResponse(ndjson_events(), media_type="application/x-ndjson")


@app.get("/api/reels/refinement-status/{job_id}", response_model=RefinementStatusResponse)
def refinement_status(request: Request, job_id: str):
    _require_community_client_identity(request)
    _enforce_rate_limit(
        request,
        "reels-refinement-status",
        limit=REELS_REFINEMENT_STATUS_RATE_LIMIT_PER_WINDOW,
    )
    with get_conn() as conn:
        job_row = fetch_one(conn, "SELECT * FROM reel_generation_jobs WHERE id = ? LIMIT 1", (job_id,))
        if not job_row:
            raise HTTPException(status_code=404, detail="job_id not found")
        if str(job_row.get("status") or "").strip().lower() == "queued":
            resumed = _resume_queued_refinement_job(conn, job_row)
            if resumed is not None:
                job_row = resumed
            else:
                refreshed = fetch_one(conn, "SELECT * FROM reel_generation_jobs WHERE id = ? LIMIT 1", (job_id,))
                if refreshed:
                    job_row = refreshed
        active_generation = _fetch_active_generation_row(
            conn,
            material_id=str(job_row.get("material_id") or ""),
            request_key=str(job_row.get("request_key") or ""),
        )
        return {
            "job_id": str(job_row.get("id") or ""),
            "status": str(job_row.get("status") or ""),
            "material_id": str(job_row.get("material_id") or ""),
            "request_key": str(job_row.get("request_key") or ""),
            "source_generation_id": str(job_row.get("source_generation_id") or ""),
            "result_generation_id": str(job_row.get("result_generation_id") or "") or None,
            "active_generation_id": str((active_generation or {}).get("id") or "") or None,
            "completed_at": str(job_row.get("completed_at") or "") or None,
            "error": str(job_row.get("error_text") or "") or None,
        }


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
        retrieval_profile="bootstrap",
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
        retrieval_profile="bootstrap",
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
def can_generate_reels(request: Request, payload: ReelsGenerateRequest):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "reels-can-generate", limit=REELS_RATE_LIMIT_PER_WINDOW)
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
            # Keep the string detail form so existing deployed clients that
            # substring-match on "material_id not found" continue to work.
            # iOS/web clients should prefer HTTP 404 + this marker over
            # localised messages when routing to recovery.
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
def can_generate_reels_any(request: Request, payload: ReelsCanGenerateAnyRequest):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "reels-can-generate-any", limit=REELS_RATE_LIMIT_PER_WINDOW)
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
    request: Request,
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
    exclude_video_ids: str = "",
):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "feed", limit=REELS_GENERATE_RATE_LIMIT_PER_WINDOW)
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

    with get_conn() as conn:
        material = fetch_all(conn, "SELECT id FROM materials WHERE id = ?", (material_id,))
        if not material:
            # Keep the string detail form so existing deployed clients that
            # substring-match on "material_id not found" continue to work.
            # iOS/web clients should prefer HTTP 404 + this marker over
            # localised messages when routing to recovery.
            raise HTTPException(status_code=404, detail="material_id not found")

        fast_mode = generation_mode == "fast"
        client_requested_slow = generation_mode == "slow"
        generation_mode_overridden = False
        if SERVERLESS_MODE:
            # Hosted/serverless: force fast retrieval profile to avoid request timeouts.
            if client_requested_slow:
                generation_mode_overridden = True
            fast_mode = True
            prefetch = min(prefetch, 6)
        effective_generation_mode = "fast" if fast_mode else "slow"
        safe_min_relevance = _normalize_min_relevance(min_relevance)
        normalized_excluded_video_ids = _parse_excluded_video_ids_param(exclude_video_ids)
        safe_video_pool_mode = _normalize_video_pool_mode(video_pool_mode)
        safe_video_duration_pref = _normalize_preferred_video_duration(preferred_video_duration)
        safe_target_clip_duration_sec, safe_target_clip_min_sec, safe_target_clip_max_sec = _resolve_target_clip_duration_bounds(
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        if SERVERLESS_MODE:
            target_total = page * limit + prefetch + 2
            sync_target_total = max(page * limit, min(target_total, page * limit + 1))
        else:
            target_total = page * limit + prefetch + (10 if fast_mode else 6)
            sync_target_total = max(page * limit, min(target_total, page * limit + (4 if fast_mode else 3)))
        request_key = _build_generation_request_key(
            material_id=material_id,
            concept_id=None,
            creative_commons_only=creative_commons_only,
            generation_mode="fast" if fast_mode else "slow",
            video_pool_mode=safe_video_pool_mode,
            preferred_video_duration=safe_video_duration_pref,
            target_clip_duration_sec=safe_target_clip_duration_sec,
            target_clip_duration_min_sec=safe_target_clip_min_sec,
            target_clip_duration_max_sec=safe_target_clip_max_sec,
        )
        active_generation = _fetch_active_generation_row(conn, material_id=material_id, request_key=request_key)
        active_generation_id = str((active_generation or {}).get("id") or "") or None
        filtered_ranked = _ranked_request_reels(
            conn,
            material_id=material_id,
            fast_mode=fast_mode,
            generation_id=active_generation_id,
            min_relevance=safe_min_relevance,
            preferred_video_duration=safe_video_duration_pref,
            target_clip_duration_sec=safe_target_clip_duration_sec,
            target_clip_duration_min_sec=safe_target_clip_min_sec,
            target_clip_duration_max_sec=safe_target_clip_max_sec,
            exclude_video_ids=normalized_excluded_video_ids,
            page=page,
            limit=limit,
        )
        response_profile = str((active_generation or {}).get("retrieval_profile") or "") or None
        refinement_job = _fetch_refinement_job_for_generation(conn, active_generation_id)
        has_visible_reels_for_page = len(filtered_ranked) > (page - 1) * limit
        if has_visible_reels_for_page:
            total = len(filtered_ranked)
            start = (page - 1) * limit
            end = start + limit
            reels = filtered_ranked[start:end]
            return {
                "page": page,
                "limit": limit,
                "total": total,
                "reels": reels,
                "generation_id": active_generation_id,
                "response_profile": response_profile,
                "refinement_job_id": str((refinement_job or {}).get("id") or "") or None,
                "refinement_status": str((refinement_job or {}).get("status") or "") or None,
                "effective_generation_mode": effective_generation_mode,
                "generation_mode_overridden": generation_mode_overridden,
            }
        # Cumulative cap: if this material already has MAX_REELS_PER_MATERIAL
        # reels on disk, don't queue another refinement. Stops unbounded
        # background generation on clients that page indefinitely. Bypassed
        # when the client isn't even asking for autofill (they're page-picking,
        # not paginating) because those requests don't produce new generations.
        material_reel_count_row = fetch_one(
            conn,
            "SELECT COUNT(*) AS reel_count FROM reels WHERE material_id = ?",
            (material_id,),
        )
        material_reel_count = int((material_reel_count_row or {}).get("reel_count") or 0)
        if autofill and material_reel_count >= MAX_REELS_PER_MATERIAL:
            logger.info(
                "feed: per-material cap reached (material_id=%s count=%d cap=%d); skipping refinement queue",
                material_id,
                material_reel_count,
                MAX_REELS_PER_MATERIAL,
            )
            refinement_job = None
        else:
            refinement_job = _queue_refinement_if_needed(
                conn,
                material_id=material_id,
                concept_id=None,
                request_key=request_key,
                generation_id=active_generation_id,
                current_reels=filtered_ranked,
                required_count=target_total if autofill else page * limit,
                generation_mode="fast" if fast_mode else "slow",
                creative_commons_only=creative_commons_only,
                preferred_video_duration=safe_video_duration_pref,
                video_pool_mode=safe_video_pool_mode,
                target_clip_duration_sec=safe_target_clip_duration_sec,
                target_clip_duration_min_sec=safe_target_clip_min_sec,
                target_clip_duration_max_sec=safe_target_clip_max_sec,
                min_relevance=safe_min_relevance,
                page_hint=page,
                page_size_hint=limit,
            )
    total = len(filtered_ranked)
    start = (page - 1) * limit
    end = start + limit
    reels = filtered_ranked[start:end]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "reels": reels,
        "generation_id": active_generation_id,
        "response_profile": response_profile,
        "refinement_job_id": str((refinement_job or {}).get("id") or "") or None,
        "refinement_status": str((refinement_job or {}).get("status") or "") or None,
        "effective_generation_mode": effective_generation_mode,
        "generation_mode_overridden": generation_mode_overridden,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest):
    _enforce_rate_limit(request, "chat", limit=CHAT_RATE_LIMIT_PER_WINDOW)
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    answer = material_intelligence_service.chat_assistant(
        message=payload.message,
        topic=payload.topic,
        text=payload.text,
        history=history,
    )
    return {"answer": answer}


@app.post("/api/reels/feedback", response_model=FeedbackResponse)
def feedback(request: Request, payload: FeedbackRequest):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "feedback", limit=FEEDBACK_RATE_LIMIT_PER_WINDOW)
    clean_reel_id = str(payload.reel_id or "").strip()
    if not clean_reel_id or len(clean_reel_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid reel_id.")
    with get_conn(transactional=True) as conn:
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
        )

    return {"status": "ok", "reel_id": clean_reel_id}


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
                active_reel_id
            FROM community_material_history
            WHERE account_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (str(account["id"]), safe_limit),
        )
    items = [_serialize_community_history_item(row) for row in rows]
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
                },
            )
    items = [_serialize_community_history_item(row) for row in normalized_items]
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
                video_pool_mode,
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
                "video_pool_mode": normalized.video_pool_mode,
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

_SIMULATION_DEFAULT_TOPICS = [
    {"subject": "Krebs Cycle", "text": "The Krebs cycle, also known as the citric acid cycle, is a series of chemical reactions in the mitochondrial matrix that oxidizes acetyl-CoA to CO2 and generates NADH, FADH2, and GTP.", "tier": "ultra-niche"},
    {"subject": "Taylor Series Convergence", "text": "A Taylor series is an infinite sum of terms calculated from the values of a function's derivatives at a single point. The radius of convergence determines where the series converges.", "tier": "ultra-niche"},
    {"subject": "CRISPR Gene Editing", "text": "CRISPR-Cas9 is a genome editing tool that uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence. It creates a double-strand break allowing for gene knockout, insertion, or correction.", "tier": "niche"},
    {"subject": "Fourier Transform", "text": "The Fourier transform decomposes a function of time into the frequencies that make it up. It transforms a signal from the time domain to the frequency domain.", "tier": "niche"},
    {"subject": "Mitosis", "text": "Mitosis is a type of cell division that produces two genetically identical daughter cells from a single parent cell. The stages are prophase, metaphase, anaphase, and telophase.", "tier": "moderate"},
    {"subject": "Quantum Entanglement", "text": "Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of one instantly influences the other, regardless of distance.", "tier": "moderate"},
    {"subject": "Game Theory", "text": "Game theory is the study of mathematical models of strategic interaction among rational agents. Key concepts include Nash equilibrium, dominant strategies, and the prisoner's dilemma.", "tier": "moderate"},
    {"subject": "World War 2", "text": "World War 2 was a global conflict from 1939 to 1945 involving the Allied and Axis powers. Key events include the invasion of Poland, D-Day, and the atomic bombings.", "tier": "broad"},
    {"subject": "Climate Change", "text": "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, especially burning fossil fuels, have been the main driver since the industrial revolution.", "tier": "broad"},
    {"subject": "Machine Learning", "text": "Machine learning is a subset of artificial intelligence where systems learn from data to improve performance on tasks without being explicitly programmed.", "tier": "broad"},
    {"subject": "Biology", "text": "Biology is the scientific study of life and living organisms. It covers topics from molecular biology and genetics to ecology and evolution.", "tier": "ultra-broad"},
    {"subject": "Mathematics", "text": "Mathematics is the abstract science of number, quantity, and space. Major branches include algebra, calculus, geometry, and statistics.", "tier": "ultra-broad"},
]

_SIMULATION_KNOWN_EDU_CHANNELS: set[str] = {
    # Math
    '3blue1brown', 'blackpenredpen', 'dr. trefor bazett', 'eddie woo',
    'mathologer', 'mindyourdecisions', 'nancypi', 'numberphile', 'patrickjmt',
    'pbs infinite series', 'professor leonard', 'stand-up maths', 'think twice',
    'tibees', 'tipping point math', 'vihart', 'zach star',
    # Physics
    'fermilab', 'minutephysics', 'physics explained', 'physics girl',
    'physics videos by eugene', 'scienceclic english', 'sixty symbols',
    'the science asylum',
    # Chemistry
    'nilered', 'nileblue', 'nurdrage', 'organic chemistry tutor',
    'periodic videos', 'the organic chemistry tutor', 'tmp chem',
    # Biology
    'amoeba sisters', 'animalogic', 'atlas pro', 'bbc earth',
    'deep look', 'journey to the microcosmos', 'mbari',
    'national geographic', 'nature on pbs', 'nick zentner', 'pbs eons',
    'stated clearly', 'tierzoo',
    # Medicine
    'chubbyemu', 'institute of human anatomy', 'ninja nerd',
    # General Science
    'alpha phoenix', 'be smart', 'domain of science', 'kyle hill',
    'minute earth', 'real science', 'sci show', 'scishow', 'scishow space',
    'smarter every day', 'smartereveryday', 'steve mould', 'the action lab',
    'up and atom', 'veritasium',
    # Space
    'anton petrov', 'astrum', 'cool worlds', 'dr. becky',
    'everyday astronaut', 'pbs space time', 'sabine hossenfelder', 'scott manley',
    # Engineering
    'engineerguy', 'engineering explained', 'practical engineering',
    'real engineering', 'the engineering mindset',
    # Electronics
    'bigclivedotcom', 'curiousmarc', 'eevblog', 'electroboom',
    'technology connections',
    # Computer Science
    'ben eater', 'computerphile', 'deepmind', 'fireship',
    'liveoverflow', 'neso academy', 'sebastian lague', 'the coding train',
    'two minute papers', 'welch labs',
    # Coding
    'freecodecamp', 'kevin powell', 'programming with mosh',
    'the net ninja', 'web dev simplified',
    # Lectures
    'bozeman science', 'crash course', 'khan academy', 'lumen learning',
    'mit opencourseware', 'professor dave explains', 'yalecourses',
    # General Explanation
    'cgp grey', 'half as interesting', 'kurzgesagt', 'lemmino',
    'reallifelore', 'ted-ed', 'tom scott', 'vox', 'vsauce', 'vsauce2',
    'wendover productions',
    # Experiments & Building
    'applied science', 'mark rober', 'stuffmadehere', 'tech ingredients',
    # History
    'extra credits', 'fall of civilizations', 'historia civilis',
    'kings and generals', 'knowledgia', 'oversimplified',
    "sam o'nella academy", 'the great war', 'the history guy',
    'timeghost history',
    # Other
    'bright side of mathematics', 'brilliant', 'captain disillusion',
    'jbstatistics', 'not just bikes', 'the royal institution',
    'simplilearn', 'dw documentary', 'frontline pbs',
    # Short aliases
    'mit', 'ted', 'pbs',
}

_SIMULATION_RED_FLAGS = frozenset([
    'reaction', 'vlog', 'unboxing', 'prank', 'challenge', 'asmr',
    'gameplay', 'lets play', 'mukbang', 'haul', 'grwm', 'top 10',
    'ranking every', 'tier list',
])


def _simulation_assess_educational(reel: dict[str, Any]) -> tuple[bool, str]:
    """Server-side educational quality scorer for simulation endpoint."""
    title = str(reel.get("video_title") or "")
    description = str(reel.get("video_description") or "")
    channel_name = str(reel.get("channel_name") or "")
    ai_summary = str(reel.get("ai_summary") or "")
    duration = int(reel.get("video_duration_sec") or 0)
    title_lower = title.lower()
    desc_lower = description.lower()
    combined = f"{title_lower} {desc_lower}"

    for flag in _SIMULATION_RED_FLAGS:
        if flag in title_lower:
            return False, f"Red flag: '{flag}' in title"

    signals = 0.0
    reasons: list[str] = []

    # Channel name (most authoritative)
    ch_lower = channel_name.lower().strip()
    if ch_lower and any(ch in ch_lower for ch in _SIMULATION_KNOWN_EDU_CHANNELS):
        signals += 1.5
        reasons.append(f"known educational channel: {channel_name}")
    elif any(ch in combined for ch in _SIMULATION_KNOWN_EDU_CHANNELS):
        signals += 1.0
        reasons.append("known educational channel (in metadata)")

    # Channel name contains educational keywords (e.g., "Physics Videos", "Geo History")
    if ch_lower and not any(ch in ch_lower for ch in _SIMULATION_KNOWN_EDU_CHANNELS):
        _edu_ch_words = ['science', 'physics', 'history', 'biology', 'chemistry',
                         'math', 'education', 'professor', 'academy', 'lecture',
                         'engineering', 'medical', 'learning', 'tutorial',
                         'university', 'institute', 'geographic']
        ch_edu_hits = sum(1 for w in _edu_ch_words if w in ch_lower)
        if ch_edu_hits >= 1:
            signals += 0.5
            reasons.append(f"edu channel name ({channel_name})")

    # Title keywords
    title_edu = ['explained', 'tutorial', 'lecture', 'lesson', 'course', 'learn',
                 'introduction', 'intro', 'how', 'what is', 'guide', 'basics',
                 'chapter', 'part', 'fundamentals', 'overview', 'crash course',
                 'made simple', 'made easy', 'for beginners', '101', 'documentary',
                 'science', 'history', 'math', 'biology', 'chemistry', 'physics',
                 'calculus', 'algebra', 'proof', 'theorem', 'derivation']
    hits = sum(1 for w in title_edu if w in title_lower)
    if hits >= 1:
        signals += 0.5 + 0.2 * min(3, hits)
        reasons.append(f"title keywords ({hits})")

    # Description keywords
    desc_edu = ['learn', 'education', 'course', 'academy', 'university', 'professor',
                'khan', 'mit', 'ted', 'lecture', 'patreon', 'practice', 'textbook',
                'curriculum', 'syllabus', 'exam', 'student', 'teaching']
    d_hits = sum(1 for w in desc_edu if w in desc_lower)
    if d_hits >= 1:
        signals += 0.3 + 0.15 * min(3, d_hits)
        reasons.append(f"desc keywords ({d_hits})")

    # Description structure
    if re.search(r'\d{1,2}:\d{2}', description):
        signals += 0.5
        reasons.append("timestamps")
    if len(description) > 500:
        signals += 0.6
        reasons.append("very detailed description")
    elif len(description) > 200:
        signals += 0.4
        reasons.append("detailed description")
    if 'http' in desc_lower:
        signals += 0.2
        reasons.append("links")

    # Title structure
    if re.search(r'\b(part|chapter|lecture|episode|module|unit)\s+\d', title_lower):
        signals += 0.5
        reasons.append("series structure")

    # Duration band
    if 300 <= duration <= 1800:
        signals += 0.3
        reasons.append(f"edu duration ({duration}s)")

    # AI summary
    if ai_summary and len(ai_summary) > 40:
        signals += 0.3
        reasons.append("has AI summary")

    # Transcript vocabulary from captions
    captions = reel.get("captions") or []
    snippet = str(reel.get("transcript_snippet") or "")
    text = " ".join(str(c.get("text", "")) for c in captions) if captions else snippet
    if text:
        words = text.split()
        if len(words) > 50:
            unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
            if unique_ratio > 0.35:
                signals += 0.5
                reasons.append(f"vocab diversity {unique_ratio:.2f}")
            signals += 0.3
            reasons.append(f"transcript ({len(words)} words)")

    is_edu = signals >= 1.0
    return is_edu, "; ".join(reasons) if reasons else "no educational signals"


def _simulation_is_sentence_start(text: str) -> bool:
    if not text.strip():
        return False
    return text.strip()[0].isupper() or text.strip()[0].isdigit() or text.strip()[0] in '"\'('


_SENTENCE_END_RE = re.compile(r"[.!?…][\"'\)\]]*\s*$")


def _simulation_is_sentence_end(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    # Accept .!?… optionally followed by closing quotes/parens (matches what
    # _refine_clip_window_from_transcript produces).  Also strip trailing
    # caption markers like [Music], (laughter), etc. before checking.
    cleaned = re.sub(r"\s*[\[\(][^\]\)]+[\]\)]\s*$", "", stripped).rstrip()
    if not cleaned:
        return False
    return bool(_SENTENCE_END_RE.search(cleaned))


@app.post("/api/admin/run-simulation", response_model=SimulationResponse)
def admin_run_simulation(request: Request, payload: AdminSimulationRequest | None = None):
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "admin-simulation", limit=ADMIN_SIMULATION_RATE_LIMIT_PER_WINDOW)

    topics = (payload.topics if payload and payload.topics else None) or _SIMULATION_DEFAULT_TOPICS
    num_reels = min(max(1, payload.num_reels if payload else 3), 5)

    results: list[SimulationTopicResult] = []
    total_clips = 0
    total_edu = 0
    total_on_topic = 0
    total_clean_start = 0
    total_clean_end = 0

    for topic in topics:
        subject = str(topic.get("subject", ""))
        text = str(topic.get("text", ""))
        tier = str(topic.get("tier", ""))
        topic_result = SimulationTopicResult(subject=subject, tier=tier)

        try:
            with get_conn() as conn:
                # Step 1: Create material
                concept_limit = 6
                concepts, objectives = material_intelligence_service.extract_concepts_and_objectives(
                    conn, text, subject_tag=subject, max_concepts=concept_limit,
                )
                material_id = str(uuid.uuid4())
                upsert(conn, "materials", {
                    "id": material_id,
                    "subject_tag": subject,
                    "raw_text": text,
                    "source_type": "text",
                    "source_path": None,
                    "created_at": now_iso(),
                })
                concept_id = ""
                subject_lower = subject.lower()
                if concepts:
                    best = concepts[0]
                    for c in concepts:
                        if subject_lower in c.get("title", "").lower():
                            best = c
                            break
                    concept_id = best.get("id", "")

                    for concept in concepts:
                        concept_text = f"{concept['title']}. Keywords: {' '.join(concept['keywords'])}. Summary: {concept['summary']}"
                        emb = embedding_service.embed_texts(conn, [concept_text])[0]
                        upsert(conn, "concepts", {
                            "id": concept["id"],
                            "material_id": material_id,
                            "title": concept["title"],
                            "keywords_json": dumps_json(concept["keywords"]),
                            "summary": concept["summary"],
                            "embedding_json": dumps_json(emb.tolist()),
                            "created_at": now_iso(),
                        })

                if not concept_id:
                    topic_result.error = "no concepts extracted"
                    results.append(topic_result)
                    continue

                # Step 2: Generate reels
                generation_result = _ensure_generation_for_request(
                    conn,
                    material_id=material_id,
                    concept_id=concept_id,
                    required_count=num_reels,
                    sync_deep_fallback="if_empty",
                    creative_commons_only=False,
                    generation_mode="fast",
                    min_relevance=None,
                    video_pool_mode="balanced",
                    preferred_video_duration="any",
                    target_clip_duration_sec=60,
                    target_clip_duration_min_sec=15,
                    target_clip_duration_max_sec=120,
                    page_hint=1,
                )

            reels = list(generation_result.get("reels") or [])[:num_reels]
            topic_result.reels_count = len(reels)

            if not reels:
                topic_result.error = "no reels generated"
                results.append(topic_result)
                continue

            # Step 3: Evaluate each reel
            keywords = text.lower().split()[:20]
            for reel in reels:
                total_clips += 1
                clip_info: dict[str, Any] = {
                    "video_title": reel.get("video_title", ""),
                    "channel_name": reel.get("channel_name", ""),
                    "t_start": reel.get("t_start", 0),
                    "t_end": reel.get("t_end", 0),
                    "video_duration_sec": reel.get("video_duration_sec", 0),
                }

                # Educational check
                is_edu, edu_notes = _simulation_assess_educational(reel)
                clip_info["educational"] = is_edu
                clip_info["edu_notes"] = edu_notes
                if is_edu:
                    total_edu += 1
                    topic_result.educational_count += 1

                # On-topic check (from captions/snippet)
                captions = reel.get("captions") or []
                snippet = str(reel.get("transcript_snippet") or "")
                clip_text = " ".join(str(c.get("text", "")) for c in captions) if captions else snippet
                if clip_text:
                    subject_words = subject.lower().split()
                    subject_hits = sum(1 for w in subject_words if w in clip_text.lower())
                    subject_cov = subject_hits / max(1, len(subject_words))
                    kw_hits = sum(1 for kw in keywords if kw in clip_text.lower())
                    kw_cov = kw_hits / max(1, len(keywords))
                    is_on_topic = subject_cov >= 0.5 or kw_cov >= 0.3
                else:
                    is_on_topic = False
                clip_info["on_topic"] = is_on_topic
                if is_on_topic:
                    total_on_topic += 1
                    topic_result.on_topic_count += 1

                # Boundary checks
                starts_clean = _simulation_is_sentence_start(clip_text) if clip_text else False
                ends_clean = _simulation_is_sentence_end(clip_text) if clip_text else False
                clip_info["starts_clean"] = starts_clean
                clip_info["ends_clean"] = ends_clean
                if starts_clean:
                    total_clean_start += 1
                    topic_result.clean_start_count += 1
                if ends_clean:
                    total_clean_end += 1
                    topic_result.clean_end_count += 1

                topic_result.clips.append(clip_info)

        except Exception as exc:
            topic_result.error = str(exc)[:300]

        results.append(topic_result)

    n = max(1, total_clips)
    return SimulationResponse(
        topics_tested=len(results),
        total_clips=total_clips,
        educational_pct=round(100.0 * total_edu / n, 1),
        on_topic_pct=round(100.0 * total_on_topic / n, 1),
        clean_start_pct=round(100.0 * total_clean_start / n, 1),
        clean_end_pct=round(100.0 * total_clean_end / n, 1),
        per_topic=results,
    )


@app.post("/api/admin/diagnose-topic", response_model=DiagnoseTopicResponse)
def admin_diagnose_topic(request: Request, payload: AdminDiagnoseTopicRequest):
    """Diagnose why a topic generates 0 reels.

    Runs the full pipeline and returns a diagnostic report showing where
    candidates are lost at each stage: search, duration filter, relevance
    gate, transcript availability.
    """
    _require_community_client_identity(request)
    _enforce_rate_limit(request, "admin-diagnose", limit=ADMIN_SIMULATION_RATE_LIMIT_PER_WINDOW)

    subject = payload.subject.strip()
    if not subject:
        return DiagnoseTopicResponse(subject=subject, error="empty subject")

    num_reels = min(max(1, payload.num_reels), 5)
    response = DiagnoseTopicResponse(subject=subject)

    try:
        with get_conn() as conn:
            # Step 1: Create material + extract concepts
            text = subject
            concepts, objectives = material_intelligence_service.extract_concepts_and_objectives(
                conn, text, subject_tag=subject, max_concepts=6,
            )
            material_id = str(uuid.uuid4())
            upsert(conn, "materials", {
                "id": material_id,
                "subject_tag": subject,
                "raw_text": text,
                "source_type": "text",
                "source_path": None,
                "created_at": now_iso(),
            })

            if not concepts:
                response.error = "no concepts extracted"
                return response

            concept = concepts[0]
            subject_lower = subject.lower()
            for c in concepts:
                if subject_lower in c.get("title", "").lower():
                    concept = c
                    break

            concept_id = concept.get("id", "")
            concept_text = f"{concept['title']}. Keywords: {' '.join(concept['keywords'])}. Summary: {concept['summary']}"
            emb = embedding_service.embed_texts(conn, [concept_text])[0]
            upsert(conn, "concepts", {
                "id": concept_id,
                "material_id": material_id,
                "title": concept["title"],
                "keywords_json": dumps_json(concept["keywords"]),
                "summary": concept["summary"],
                "embedding_json": dumps_json(emb.tolist()),
                "created_at": now_iso(),
            })

            # Step 2: Generate reels with diagnostic tracking
            generation_result = _ensure_generation_for_request(
                conn,
                material_id=material_id,
                concept_id=concept_id,
                required_count=num_reels,
                sync_deep_fallback="if_empty",
                creative_commons_only=False,
                generation_mode="fast",
                min_relevance=None,
                video_pool_mode="balanced",
                preferred_video_duration="any",
                target_clip_duration_sec=60,
                target_clip_duration_min_sec=15,
                target_clip_duration_max_sec=120,
                page_hint=1,
            )

            reels = list(generation_result.get("reels") or [])[:num_reels]
            response.final_reel_count = len(reels)

            # Step 3: Collect diagnostic data from DB
            # Search queries used (search_cache may not exist on all schemas — fail open)
            try:
                search_rows = fetch_all(
                    conn,
                    "SELECT query FROM search_cache WHERE query LIKE ? ORDER BY created_at DESC LIMIT 20",
                    (f"%{subject_lower[:20]}%",),
                )
                response.queries_generated = [str(r.get("query", "")) for r in search_rows]
            except Exception:
                response.queries_generated = []

            reel_stage = DiagnosticStageResult(
                stage="reels_created",
                candidate_count=len(reels),
                details=[
                    {
                        "video_title": r.get("video_title", ""),
                        "channel_name": r.get("channel_name", ""),
                        "t_start": r.get("t_start", 0),
                        "t_end": r.get("t_end", 0),
                        "score": r.get("score", 0),
                        "source_surface": r.get("source_surface", ""),
                        "relevance_score": r.get("relevance_score", 0),
                    }
                    for r in reels
                ],
            )
            response.stages.append(reel_stage)

            # Transcript availability
            transcript_stage = DiagnosticStageResult(stage="transcript_availability")
            for reel in reels:
                vid = reel.get("video_id", "")
                if vid:
                    tq = youtube_service.get_transcript_quality(conn, vid)
                    transcript_stage.details.append({
                        "video_id": vid,
                        "has_transcript": tq is not None,
                        "source_kind": tq.get("source_kind") if tq else None,
                        "quality_score": tq.get("quality_score") if tq else None,
                        "quality_status": tq.get("quality_status") if tq else None,
                    })
            transcript_stage.candidate_count = sum(1 for d in transcript_stage.details if d.get("has_transcript"))
            response.stages.append(transcript_stage)

            # Surface retrieval_metrics + per-segment branch counts from the
            # most recent retrieval_runs entries for this material. These rows
            # are written by `_persist_retrieval_debug_run` when
            # retrieval_debug_logging is on (default true). Aggregating across
            # all concept runs for this material gives visibility into WHERE
            # in the segment loop candidates drop off.
            try:
                run_rows = fetch_all(
                    conn,
                    "SELECT id, concept_id, concept_title, selected_video_id, "
                    "failure_reason, debug_json, created_at "
                    "FROM retrieval_runs WHERE material_id = ? "
                    "ORDER BY created_at DESC LIMIT 10",
                    (material_id,),
                )
                aggregated: dict[str, Any] = {}
                run_summaries: list[dict] = []
                for row in run_rows:
                    debug_json_raw = row.get("debug_json") or "{}"
                    try:
                        if isinstance(debug_json_raw, (bytes, bytearray)):
                            debug_json_raw = debug_json_raw.decode("utf-8", errors="ignore")
                        parsed = (
                            debug_json_raw
                            if isinstance(debug_json_raw, dict)
                            else json.loads(str(debug_json_raw))
                        )
                    except Exception:
                        parsed = {}
                    metrics = parsed.get("metrics") or {}
                    run_summaries.append({
                        "run_id": str(row.get("id") or ""),
                        "concept_id": str(row.get("concept_id") or ""),
                        "concept_title": str(row.get("concept_title") or ""),
                        "selected_video_id": str(row.get("selected_video_id") or ""),
                        "failure_reason": str(row.get("failure_reason") or ""),
                        "candidate_count": int(parsed.get("candidate_count") or 0),
                        "query_count": int(parsed.get("query_count") or 0),
                        "metrics": metrics,
                    })
                    for key, value in metrics.items():
                        try:
                            aggregated[key] = int(aggregated.get(key, 0)) + int(value or 0)
                        except (TypeError, ValueError):
                            aggregated[key] = value
                response.retrieval_metrics = aggregated
                response.retrieval_runs = run_summaries
            except Exception:
                logger.exception("diagnose-topic: failed to load retrieval_runs for %s", material_id)

            if not reels:
                response.error = f"0 reels generated for '{subject}' — check stages for where candidates were lost"

    except Exception as exc:
        response.error = str(exc)[:500]

    return response


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
