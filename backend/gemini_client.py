"""Gemini (Google AI Studio) wrapper for structured JSON selection output.

Uses the google-genai SDK's `models.generate_content` with `response_mime_type=
application/json` + a Pydantic `response_schema`, which enforces the output shape.
Includes a simple retry on rate-limit / transient errors.
"""
from __future__ import annotations

import math
import random
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Optional

import httpx
from google import genai
from google.genai import errors as genai_errors, types

from . import config

_client: Optional[genai.Client] = None
_lock = threading.Lock()


@dataclass(frozen=True)
class GeminiCallTelemetry:
    """Provider metadata for one logical Gemini structured-content call."""

    model: str
    operation: str
    prompt_version: str
    thinking_level: str
    latency_ms: float
    retries: int
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    candidate_tokens: Optional[int]
    thought_tokens: Optional[int]
    total_tokens: Optional[int]
    cached_tokens: Optional[int] = None
    provider_error_type: Optional[str] = None
    provider_status_code: Optional[int] = None
    retryable: Optional[bool] = None
    error_history: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


class _TokenCountResponseError(RuntimeError):
    """A successful CountTokens response was malformed or incomplete."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    telemetry: GeminiCallTelemetry


class GeminiCallError(RuntimeError):
    """A typed Gemini failure carrying telemetry collected before failure."""

    def __init__(self, message: str, telemetry: GeminiCallTelemetry):
        super().__init__(message)
        self.telemetry = telemetry


class GeminiEmptyResponseError(GeminiCallError):
    pass


class GeminiTruncatedResponseError(GeminiCallError):
    pass


class GeminiBlockedResponseError(GeminiCallError):
    pass


class GeminiDeadlineExceededError(GeminiCallError):
    pass


class GeminiCancelledError(GeminiCallError):
    pass


class GeminiTransportError(GeminiCallError):
    pass


def _reject_video_content(value: object) -> None:
    """Fail before dispatch when any Gemini request contains video media."""
    if isinstance(value, Mapping):
        normalized = {
            str(key).replace("_", "").casefold(): item
            for key, item in value.items()
        }
        mime_type = str(normalized.get("mimetype") or "").casefold()
        file_uri = str(normalized.get("fileuri") or "").casefold()
        if (
            mime_type.startswith("video/")
            or normalized.get("videometadata") is not None
            or "youtube.com/" in file_uri
            or "youtu.be/" in file_uri
            or re.search(
                r"\.(?:m4v|mkv|mov|mp4|webm)(?:$|[?#])", file_uri,
            )
        ):
            raise ValueError("Gemini video input is disabled")
        for item in value.values():
            _reject_video_content(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _reject_video_content(item)
        return
    content_parts = getattr(value, "parts", None)
    if isinstance(content_parts, (list, tuple)):
        _reject_video_content(content_parts)
    inline_data = getattr(value, "inline_data", None)
    inline_mime = str(getattr(inline_data, "mime_type", "") or "").casefold()
    file_data = getattr(value, "file_data", None)
    file_mime = str(getattr(file_data, "mime_type", "") or "").casefold()
    file_uri = str(getattr(file_data, "file_uri", "") or "").casefold()
    direct_mime = str(getattr(value, "mime_type", "") or "").casefold()
    direct_uri = str(getattr(value, "file_uri", "") or "").casefold()
    video_metadata = getattr(value, "video_metadata", None)
    if (
        inline_mime.startswith("video/")
        or file_mime.startswith("video/")
        or direct_mime.startswith("video/")
        or video_metadata is not None
        or "youtube.com/" in file_uri
        or "youtu.be/" in file_uri
        or "youtube.com/" in direct_uri
        or "youtu.be/" in direct_uri
        or re.search(r"\.(?:m4v|mkv|mov|mp4|webm)(?:$|[?#])", file_uri)
        or re.search(r"\.(?:m4v|mkv|mov|mp4|webm)(?:$|[?#])", direct_uri)
    ):
        raise ValueError("Gemini video input is disabled")


def get_client() -> genai.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                if not config.GEMINI_API_KEY:
                    raise RuntimeError(
                        "GEMINI_API_KEY is not set. Add it to .env "
                        "(get a free key at https://aistudio.google.com/apikey)."
                    )
                _client = genai.Client(
                    api_key=config.GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=120_000),  # ms; don't hang forever
                )
    return _client


def _is_retryable(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "429" in s or "resource_exhausted" in s or "rate" in s
        or "503" in s or "500" in s or "unavailable" in s or "overloaded" in s
    )


def _retry(call) -> str:
    """Run ``call`` (returns response text or falsy) with backoff on transient errors."""
    last: Optional[Exception] = None
    for attempt in range(config.BACKOFF_MAX_RETRIES + 1):
        try:
            text = call()
            if text:
                return text
            last = RuntimeError("empty response from Gemini")
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_retryable(e) or attempt == config.BACKOFF_MAX_RETRIES:
                raise
        wait = min(config.BACKOFF_CAP, config.BACKOFF_BASE * 2 ** attempt)
        time.sleep(wait + random.uniform(0, 0.5 * wait))
    raise RuntimeError(f"Gemini call failed: {last}")


_GEMINI3_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_gemini3_model(model: str) -> bool:
    name = str(model or "").rsplit("/", 1)[-1].lower()
    return name == "gemini-3" or name.startswith(("gemini-3.", "gemini-3-"))


def _default_gemini3_thinking_level(model: str) -> str:
    return "high" if "pro" in str(model or "").lower() else "medium"


def _field(value, name: str):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def count_request_tokens(
    system: str,
    user_text: str,
    schema,
    *,
    model: str,
    timeout_s: float = 10.0,
    thinking_level: str = "",
    max_output_tokens: int | None = None,
    deadline_monotonic: float | None = None,
    cancelled=None,
) -> int:
    """Count the exact structured request with one bounded transient retry."""
    mdl = str(model or "").strip()
    timeout = float(timeout_s)
    if not _is_gemini3_model(mdl):
        raise ValueError("count_request_tokens requires an explicit Gemini 3 model")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_s must be positive")
    if (
        deadline_monotonic is not None
        and not math.isfinite(float(deadline_monotonic))
    ):
        raise ValueError("deadline_monotonic must be finite")
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    model_name = mdl.rsplit("/", 1)[-1]
    generation_config: dict[str, object] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": _gemini3_json_schema(schema),
    }
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max(1, int(max_output_tokens))
    level = str(thinking_level or "").strip().upper()
    if level:
        if level.casefold() not in _GEMINI3_THINKING_LEVELS:
            raise ValueError(f"unsupported Gemini 3 thinking level: {thinking_level}")
        generation_config["thinkingConfig"] = {"thinkingLevel": level}
    request_body = {
        "generateContentRequest": {
            "model": f"models/{model_name}",
            "contents": [{
                "role": "user",
                "parts": [{"text": str(user_text or "")}],
            }],
            "systemInstruction": {
                "role": "user",
                "parts": [{"text": str(system or "")}],
            },
            "generationConfig": generation_config,
        }
    }
    operation_deadline = min(
        time.monotonic() + timeout,
        (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else math.inf
        ),
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:countTokens"
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": config.GEMINI_API_KEY,
    }

    for attempt in range(2):
        if _cancel_requested(cancelled):
            raise RuntimeError("Gemini token count cancelled")
        remaining_s = operation_deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError("Gemini token count deadline exceeded")
        request_timeout_s = (
            timeout
            if attempt == 0 and deadline_monotonic is None
            else min(timeout, remaining_s)
        )
        try:
            response = httpx.post(
                url,
                headers=headers,
                json=request_body,
                timeout=request_timeout_s,
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except (TypeError, ValueError) as exc:
                raise _TokenCountResponseError(
                    "Gemini token count response was not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise _TokenCountResponseError(
                    "Gemini token count response was not an object"
                )
            try:
                total = int(_field(payload, "totalTokens") or 0)
            except (TypeError, ValueError, OverflowError) as exc:
                raise _TokenCountResponseError(
                    "Gemini token count response had an invalid total"
                ) from exc
            if total <= 0:
                raise _TokenCountResponseError(
                    "Gemini token count response omitted the total"
                )
            return total
        except Exception as error:  # noqa: BLE001
            if _cancel_requested(cancelled):
                raise RuntimeError("Gemini token count cancelled") from error
            if attempt > 0 or not _transient_token_count_error(error):
                raise
            wait_s = max(
                random.uniform(0.25, 0.5),
                _gemini_retry_after(error) or 0.0,
            )
            if operation_deadline - time.monotonic() <= wait_s:
                raise TimeoutError(
                    "Gemini token count deadline leaves no retry window"
                ) from error
            if not _sleep_before_retry(wait_s, cancelled):
                raise RuntimeError("Gemini token count cancelled") from error

    raise AssertionError("unreachable Gemini token-count retry state")


def _finish_reason(response) -> Optional[str]:
    candidates = _field(response, "candidates") or []
    if not candidates:
        return None
    reason = _field(candidates[0], "finish_reason")
    if reason is None:
        return None
    return str(getattr(reason, "value", reason))


def _call_telemetry(*, model: str, operation: str, prompt_version: str,
                    thinking_level: str, started: float, retries: int,
                    response=None, error: Exception | None = None,
                    retryable: bool | None = None,
                    error_history=()) -> GeminiCallTelemetry:
    usage = _field(response, "usage_metadata") if response is not None else None
    return GeminiCallTelemetry(
        model=model,
        operation=operation,
        prompt_version=prompt_version,
        thinking_level=thinking_level,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        retries=retries,
        finish_reason=_finish_reason(response) if response is not None else None,
        prompt_tokens=_field(usage, "prompt_token_count"),
        candidate_tokens=_field(usage, "candidates_token_count"),
        thought_tokens=_field(usage, "thoughts_token_count"),
        total_tokens=_field(usage, "total_token_count"),
        cached_tokens=_field(usage, "cached_content_token_count"),
        provider_error_type=type(error).__name__ if error is not None else None,
        provider_status_code=(
            _gemini_status_code(error) if error is not None else None
        ),
        retryable=retryable if error is not None else None,
        error_history=tuple(dict(item) for item in error_history),
    )


def _gemini_status_code(error: Exception) -> int | None:
    """Extract a typed HTTP status without relying on provider error prose."""
    response = getattr(error, "response", None)
    for raw_status in (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(response, "status_code", None),
    ):
        status = getattr(raw_status, "value", raw_status)
        try:
            return int(status)
        except (TypeError, ValueError):
            continue
    return None


def _transient_gemini_error(
    error: Exception,
    *,
    retry_provider_cancelled: bool = False,
) -> bool:
    status = _gemini_status_code(error)
    if status is not None:
        # Google can return 499 when its serving attempt is cancelled even
        # though the application is still active. Treat that provider-side
        # cancellation as transient only for callers that have explicitly
        # confirmed their own cancellation signal is clear.
        if status == 499:
            return bool(retry_provider_cancelled)
        # A known status is authoritative. In particular, do not let text such
        # as "field unavailable" turn a 4xx schema/auth failure into a retry.
        return status in _TRANSIENT_STATUS_CODES
    if isinstance(error, genai_errors.ClientError):
        return False
    message = str(error).lower()
    if any(marker in message for marker in (
        "invalid_argument", "unauthenticated", "permission_denied",
        "api key", "response schema", "json schema",
    )):
        return False
    if isinstance(error, (httpx.TransportError, httpx.DecodingError)):
        return True
    if isinstance(error, genai_errors.UnknownApiResponseError):
        # The SDK received a successful but malformed/truncated provider body.
        return True
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    name = type(error).__name__.lower()
    if any(marker in name for marker in ("timeout", "connect", "network", "protocol")):
        return True
    return any(
        marker in message
        for marker in (
            "resource_exhausted", "unavailable", "overloaded",
            "server disconnected", "connection reset", "connection closed",
            "peer closed",
            "status 408", "status 429", "status 500", "status 502",
            "status 503", "status 504",
        )
    )


def _transient_token_count_error(error: Exception) -> bool:
    """Classify CountTokens transport failures without retrying ordinary 4xx."""
    if isinstance(error, _TokenCountResponseError):
        return True
    status = _gemini_status_code(error)
    if status is not None:
        return status in {408, 429} or 500 <= status <= 599
    return _transient_gemini_error(error)


def _gemini_retry_after(error: Exception, *, maximum: float = 30.0) -> float | None:
    """Return a bounded provider Retry-After delay without retaining error prose."""
    response = getattr(error, "response", None)
    headers = getattr(error, "headers", None) or getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw).strip())
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = (target - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds):
        return None
    return max(0.0, min(float(maximum), seconds))


def _cancel_requested(cancelled) -> bool:
    if cancelled is None:
        return False
    if callable(cancelled):
        return bool(cancelled())
    is_set = getattr(cancelled, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancelled)


def _sleep_before_retry(seconds: float, cancelled) -> bool:
    """Wait for a retry while polling the generation cancellation callback."""
    if cancelled is None:
        time.sleep(seconds)
        return True
    wait_deadline = time.monotonic() + seconds
    while True:
        if _cancel_requested(cancelled):
            return False
        remaining_s = wait_deadline - time.monotonic()
        if remaining_s <= 0:
            return True
        time.sleep(min(0.05, remaining_s))


_GEMINI3_JSON_SCHEMA_KEYS = {
    "$id", "$defs", "$ref", "$anchor", "type", "format", "title",
    "description", "enum", "items", "prefixItems", "minItems", "minimum",
    "maximum", "anyOf", "oneOf", "properties", "required", "propertyOrdering",
}


def _gemini3_json_schema(schema) -> dict:
    """Return the JSON-Schema subset accepted by Gemini 3 GenerateContent.

    Gemini 3.5 currently rejects Pydantic's legacy ``response_schema`` form for
    nested models and rejects some otherwise-standard validation keywords. The
    omitted constraints are still enforced when the caller validates the JSON
    with the original Pydantic model.
    """
    if isinstance(schema, dict):
        raw = schema
    elif hasattr(schema, "model_json_schema"):
        raw = schema.model_json_schema()
    else:
        from pydantic import TypeAdapter

        raw = TypeAdapter(schema).json_schema()

    def supported(value, *, names: bool = False):
        if isinstance(value, list):
            return [supported(item) for item in value]
        if not isinstance(value, dict):
            return value
        if names:
            return {name: supported(child) for name, child in value.items()}
        cleaned = {}
        for key, child in value.items():
            if key not in _GEMINI3_JSON_SCHEMA_KEYS:
                continue
            cleaned[key] = supported(
                child, names=key in {"properties", "$defs"},
            )
        return cleaned

    return supported(raw)


def generate_json_v3(
    system: str,
    user: str | list,
    schema,
    *,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
    timeout_s: float,
    deadline_monotonic: Optional[float],
    operation: str,
    prompt_version: str,
    initial_attempt_deadline_monotonic: Optional[float] = None,
    max_retries: int = 1,
    use_full_transient_retry_budget: bool = False,
    retry_status_codes: frozenset[int] | set[int] | None = None,
    cancelled=None,
    media_resolution=None,
    before_dispatch: Callable[..., object] | None = None,
    after_dispatch: Callable[..., object] | None = None,
) -> GenerationResult:
    """Run one Gemini 3 structured call with bounded transport behavior.

    SDK retries are disabled so ``max_retries`` is the application-controlled
    retry ceiling. ``retry_status_codes`` optionally narrows typed HTTP retries
    for latency-sensitive callers; the default preserves the general transient
    policy. A second retry is permitted only after an HTTP 503; non-503 failures
    remain capped at one retry. ``deadline_monotonic`` is an absolute
    ``time.monotonic()`` deadline shared by the caller's complete workflow.
    ``initial_attempt_deadline_monotonic`` may reserve the workflow tail while
    still allowing a transient retry to use the full operation deadline.
    Cancellation is cooperative between in-flight HTTP requests. Optional
    dispatch hooks run once around every physical provider attempt, including
    retries, so callers can admit and settle attempt-scoped resources.
    """
    _reject_video_content(user)
    mdl = str(model or "").strip()
    level = str(thinking_level or "").strip().lower()
    if not _is_gemini3_model(mdl):
        raise ValueError("generate_json_v3 requires an explicit Gemini 3 model")
    if level not in _GEMINI3_THINKING_LEVELS:
        raise ValueError(f"Unsupported Gemini 3 thinking level: {thinking_level!r}")
    if int(max_output_tokens) <= 0:
        raise ValueError("max_output_tokens must be positive")
    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise ValueError("timeout_s must be positive")
    if deadline_monotonic is not None and not math.isfinite(float(deadline_monotonic)):
        raise ValueError("deadline_monotonic must be finite")
    if (
        initial_attempt_deadline_monotonic is not None
        and not math.isfinite(float(initial_attempt_deadline_monotonic))
    ):
        raise ValueError("initial_attempt_deadline_monotonic must be finite")
    if (isinstance(max_retries, bool) or not isinstance(max_retries, int)
            or max_retries not in (0, 1, 2)):
        raise ValueError("max_retries must be 0, 1, or 2")
    if retry_status_codes is None:
        allowed_retry_statuses = None
    elif (
        not isinstance(retry_status_codes, (set, frozenset))
        or not retry_status_codes
        or any(
            isinstance(status, bool)
            or not isinstance(status, int)
            or status not in _TRANSIENT_STATUS_CODES
            for status in retry_status_codes
        )
    ):
        raise ValueError("retry_status_codes must contain transient HTTP statuses")
    else:
        allowed_retry_statuses = frozenset(retry_status_codes)

    started = time.perf_counter()
    operation_deadline = (
        float(deadline_monotonic)
        if deadline_monotonic is not None
        else time.monotonic() + float(timeout_s)
    )
    initial_attempt_deadline = (
        min(operation_deadline, float(initial_attempt_deadline_monotonic))
        if initial_attempt_deadline_monotonic is not None
        else operation_deadline
    )
    client = None
    max_attempts = max_retries + 1
    requests_started = 0
    error_history: list[dict[str, object]] = []
    last_failure_telemetry: GeminiCallTelemetry | None = None

    for attempt in range(max_attempts):
        if _cancel_requested(cancelled):
            telemetry = _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
                error_history=error_history,
            )
            raise GeminiCancelledError("Gemini call cancelled", telemetry)

        attempt_deadline = (
            initial_attempt_deadline if attempt == 0 else operation_deadline
        )
        remaining_s = attempt_deadline - time.monotonic()
        if remaining_s <= 0 or (attempt > 0 and remaining_s < 5.0):
            telemetry = last_failure_telemetry or _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
                error_history=error_history,
            )
            raise GeminiDeadlineExceededError("Gemini call deadline exceeded", telemetry)
        if client is None:
            client = get_client()
        # Client construction can block on credential or transport setup. Do
        # not acquire an attempt ticket if the call became ineligible before a
        # provider dispatch was possible.
        if _cancel_requested(cancelled):
            telemetry = _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
                error_history=error_history,
            )
            raise GeminiCancelledError("Gemini call cancelled", telemetry)
        if time.monotonic() >= attempt_deadline:
            telemetry = last_failure_telemetry or _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
                error_history=error_history,
            )
            raise GeminiDeadlineExceededError(
                "Gemini call deadline exceeded", telemetry,
            )

        dispatch_ticket = None
        if before_dispatch is not None:
            dispatch_ticket = before_dispatch(model=mdl, attempt=attempt + 1)
        application_cancelled = _cancel_requested(cancelled)
        if application_cancelled or time.monotonic() >= attempt_deadline:
            telemetry = last_failure_telemetry or _call_telemetry(
                model=mdl,
                operation=operation,
                prompt_version=prompt_version,
                thinking_level=level,
                started=started,
                retries=max(0, requests_started - 1),
                error_history=error_history,
            )
            if after_dispatch is not None:
                after_dispatch(
                    dispatch_ticket,
                    model=mdl,
                    attempt=attempt + 1,
                    telemetry=telemetry,
                    dispatched=False,
                )
            if application_cancelled:
                raise GeminiCancelledError("Gemini call cancelled", telemetry)
            raise GeminiDeadlineExceededError(
                "Gemini call deadline exceeded", telemetry,
            )
        try:
            request_timeout_s = min(
                float(timeout_s),
                attempt_deadline - time.monotonic(),
            )
            http_options = types.HttpOptions(
                timeout=max(1, int(request_timeout_s * 1000.0)),
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            config_kwargs = dict(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=_gemini3_json_schema(schema),
                max_output_tokens=int(max_output_tokens),
                thinking_config=types.ThinkingConfig(thinking_level=level),
                http_options=http_options,
            )
            if media_resolution is not None:
                config_kwargs["media_resolution"] = media_resolution
            request_config = types.GenerateContentConfig(**config_kwargs)
        except Exception as error:  # noqa: BLE001
            telemetry = _call_telemetry(
                model=mdl,
                operation=operation,
                prompt_version=prompt_version,
                thinking_level=level,
                started=started,
                retries=max(0, requests_started - 1),
                error=error,
                error_history=error_history,
            )
            if after_dispatch is not None:
                after_dispatch(
                    dispatch_ticket,
                    model=mdl,
                    attempt=attempt + 1,
                    telemetry=telemetry,
                    dispatched=False,
                )
            raise
        requests_started += 1
        try:
            response = client.models.generate_content(
                model=mdl, contents=user, config=request_config,
            )
        except Exception as error:  # noqa: BLE001
            application_cancelled = _cancel_requested(cancelled)
            retryable = _transient_gemini_error(
                error,
                retry_provider_cancelled=not application_cancelled,
            )
            error_history.append({
                "provider_error_type": type(error).__name__,
                "provider_status_code": _gemini_status_code(error),
                "retryable": retryable,
            })
            telemetry = _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
                error=error,
                retryable=retryable,
                error_history=error_history,
            )
            last_failure_telemetry = telemetry
            if after_dispatch is not None:
                after_dispatch(
                    dispatch_ticket,
                    model=mdl,
                    attempt=attempt + 1,
                    telemetry=telemetry,
                )
            if application_cancelled or _cancel_requested(cancelled):
                raise GeminiCancelledError("Gemini call cancelled", telemetry) from error
            if time.monotonic() >= operation_deadline:
                raise GeminiDeadlineExceededError(
                    "Gemini call deadline exceeded", telemetry,
                ) from error
            # The production selector may request one extra attempt for a
            # short-lived Gemini capacity spike. Other transient failures keep
            # the original one-retry ceiling so quota and latency cannot grow.
            status_code = _gemini_status_code(error)
            if (
                allowed_retry_statuses is not None
                and status_code not in allowed_retry_statuses
            ):
                retry_limit = 0
            else:
                retry_limit = (
                    max_retries
                    if use_full_transient_retry_budget or status_code == 503
                    else min(max_retries, 1)
                )
            if not retryable or attempt >= retry_limit:
                raise GeminiTransportError(str(error), telemetry) from error

            delay_floor_s = 2.0 ** attempt
            wait_s = max(
                random.uniform(delay_floor_s, delay_floor_s * 2.0),
                _gemini_retry_after(error) or 0.0,
            )
            remaining_after_wait_s = operation_deadline - time.monotonic() - wait_s
            if remaining_after_wait_s < 5.0:
                raise GeminiDeadlineExceededError(
                    "Gemini call deadline leaves no useful retry window", telemetry,
                ) from error
            if not _sleep_before_retry(wait_s, cancelled):
                raise GeminiCancelledError("Gemini call cancelled", telemetry) from error
            if operation_deadline - time.monotonic() < 5.0:
                raise GeminiDeadlineExceededError(
                    "Gemini call deadline leaves no useful retry window", telemetry,
                ) from error
            continue

        telemetry = _call_telemetry(
            model=mdl, operation=operation, prompt_version=prompt_version,
            thinking_level=level, started=started,
            retries=max(0, requests_started - 1), response=response,
            error_history=error_history,
        )
        if after_dispatch is not None:
            after_dispatch(
                dispatch_ticket,
                model=mdl,
                attempt=attempt + 1,
                telemetry=telemetry,
            )
        if _cancel_requested(cancelled):
            raise GeminiCancelledError("Gemini call cancelled", telemetry)
        if time.monotonic() >= attempt_deadline:
            raise GeminiDeadlineExceededError("Gemini call deadline exceeded", telemetry)
        finish_reason = (telemetry.finish_reason or "").upper()
        response_error_type: type[GeminiCallError] | None = None
        response_error_message = ""
        if finish_reason.endswith("MAX_TOKENS"):
            response_error_type = GeminiTruncatedResponseError
            response_error_message = "Gemini response reached max_output_tokens"
        elif finish_reason and not finish_reason.endswith("STOP"):
            raise GeminiBlockedResponseError(
                f"Gemini response did not finish normally ({telemetry.finish_reason})",
                telemetry,
            )
        text = None
        if response_error_type is None:
            try:
                text = response.text
            except Exception:  # noqa: BLE001 - invalid provider response
                text = None
            if not text or not str(text).strip():
                response_error_type = GeminiEmptyResponseError
                response_error_message = "empty response from Gemini"
        if response_error_type is not None:
            error_history.append({
                "provider_error_type": response_error_type.__name__,
                "provider_status_code": None,
                "retryable": True,
            })
            telemetry = _call_telemetry(
                model=mdl,
                operation=operation,
                prompt_version=prompt_version,
                thinking_level=level,
                started=started,
                retries=max(0, requests_started - 1),
                response=response,
                retryable=True,
                error_history=error_history,
            )
            last_failure_telemetry = telemetry
            if _cancel_requested(cancelled):
                raise GeminiCancelledError("Gemini call cancelled", telemetry)
            if attempt >= max_retries:
                raise response_error_type(response_error_message, telemetry)
            if operation_deadline - time.monotonic() < 5.0:
                raise GeminiDeadlineExceededError(
                    "Gemini call deadline leaves no useful retry window",
                    telemetry,
                )
            continue
        assert text is not None
        return GenerationResult(text=str(text), telemetry=telemetry)

    raise AssertionError("unreachable Gemini retry state")


def generate_json(system: str, user: str, schema, temperature: float = 0.2,
                  model: Optional[str] = None, max_output_tokens: int = 8192,
                  thinking_level: Optional[str] = None) -> str:
    """Return a JSON string conforming to `schema` (a Pydantic model class).

    ``max_output_tokens`` bounds the response; raise it for calls whose JSON is large
    or that use a thinking model because thinking tokens share the same budget. Gemini 3
    uses a thinking level; the legacy Gemini 2.5 path retains its thinking-budget behavior.
    """
    _reject_video_content(user)
    mdl = model or config.GEMINI_MODEL

    if _is_gemini3_model(mdl):
        level = thinking_level or _default_gemini3_thinking_level(mdl)
        return generate_json_v3(
            system,
            user,
            schema,
            model=mdl,
            thinking_level=level,
            max_output_tokens=max_output_tokens,
            timeout_s=120.0,
            deadline_monotonic=None,
            operation="generate_json",
            prompt_version="legacy",
            max_retries=1,
        ).text

    client = get_client()

    def _make_config(thinking: bool):
        kwargs = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if not thinking:
            # Disable 2.5-flash "thinking" → much faster for structured extraction.
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    def _call():
        try:
            resp = client.models.generate_content(
                model=mdl, contents=user, config=_make_config(thinking=False)
            )
        except Exception:
            # some models reject thinking_budget=0 → retry with thinking on
            resp = client.models.generate_content(
                model=mdl, contents=user, config=_make_config(thinking=True)
            )
        return resp.text

    return _retry(_call)


# ── multimodal (Phase 2 vision) ──────────────────────────────────────────────
def image_part(jpeg_bytes: bytes):
    """A Gemini image Part from raw JPEG bytes."""
    return types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")


def text_part(text: str):
    """A Gemini text Part."""
    return types.Part.from_text(text=text)


def youtube_video_part(url: str, *, end_offset_sec: float | None = None):
    """Retired: production never sends YouTube video to Gemini."""
    del url, end_offset_sec
    raise ValueError("Gemini video input is disabled")


# ── video (Wave 4 edge-probe / render-audit) ─────────────────────────────────
_MEDIA_RES = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}


def media_resolution_from_name(name):
    """Map a config string ('low'|'medium'|'high') → a ``types.MediaResolution`` enum
    (LOW ≈ 100 tok/s of video, the cost default). Unknown/empty → None (SDK default)."""
    return _MEDIA_RES.get(str(name or "").strip().lower())


def video_part_inline(data: bytes, media_resolution=None):
    """Retired: production never sends inline video to Gemini."""
    del data, media_resolution
    raise ValueError("Gemini video input is disabled")


def generate_json_video(system: str, parts: list, schema, media_resolution=None,
                        temperature: float = 0.2, model: Optional[str] = None) -> str:
    """Retired: production never sends video to Gemini."""
    del system, parts, schema, media_resolution, temperature, model
    raise ValueError("Gemini video input is disabled")


def generate_json_mm(system: str, parts: list, schema, temperature: float = 0.2) -> str:
    """Return a JSON string conforming to ``schema`` from a multimodal prompt.

    ``parts`` is a list of ``types.Part`` (interleaved text + image bytes), sent as the
    single user turn. Gemini 2.5 retains the legacy thinking-budget behavior; Gemini 3
    uses a thinking level and default sampling.
    """
    _reject_video_content(parts)
    if _is_gemini3_model(config.GEMINI_MODEL):
        return generate_json_v3(
            system,
            parts,
            schema,
            model=config.GEMINI_MODEL,
            thinking_level=_default_gemini3_thinking_level(config.GEMINI_MODEL),
            max_output_tokens=8192,
            timeout_s=120.0,
            deadline_monotonic=None,
            operation="generate_json_mm",
            prompt_version="legacy",
            max_retries=1,
        ).text

    client = get_client()

    def _make_config(thinking: bool):
        kwargs = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            max_output_tokens=8192,
        )
        if not thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    def _call():
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL, contents=parts, config=_make_config(thinking=False)
            )
        except Exception:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL, contents=parts, config=_make_config(thinking=True)
            )
        return resp.text

    return _retry(_call)
