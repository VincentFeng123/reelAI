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
from dataclasses import asdict, dataclass
from typing import Optional

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

    def as_dict(self) -> dict:
        return asdict(self)


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
                    retryable: bool | None = None) -> GeminiCallTelemetry:
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
    )


def _gemini_status_code(error: Exception) -> int | None:
    """Extract an HTTP status without relying on provider error prose."""
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

    message = str(error)
    match = re.match(r"\s*(\d{3})\b", message)
    if match is None:
        match = re.search(
            r"\b(?:http(?:\s+status)?|status|code)\s*[:=]?\s*(\d{3})\b",
            message,
            re.IGNORECASE,
        )
    return int(match.group(1)) if match is not None else None


def _transient_gemini_error(error: Exception) -> bool:
    status = _gemini_status_code(error)
    if status is not None:
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


def _cancel_requested(cancelled) -> bool:
    if cancelled is None:
        return False
    if callable(cancelled):
        return bool(cancelled())
    is_set = getattr(cancelled, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(cancelled)


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
    user: str,
    schema,
    *,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
    timeout_s: float,
    deadline_monotonic: Optional[float],
    operation: str,
    prompt_version: str,
    max_retries: int = 1,
    cancelled=None,
    media_resolution=None,
) -> GenerationResult:
    """Run one Gemini 3 structured call with bounded transport behavior.

    SDK retries are disabled so ``max_retries=1`` means at most one
    application-controlled retry. ``deadline_monotonic`` is an absolute
    ``time.monotonic()`` deadline shared by the caller's complete workflow.
    Cancellation is cooperative between in-flight HTTP requests.
    """
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
    if (isinstance(max_retries, bool) or not isinstance(max_retries, int)
            or max_retries not in (0, 1)):
        raise ValueError("max_retries must be 0 or 1")

    started = time.perf_counter()
    operation_deadline = time.monotonic() + float(timeout_s)
    client = None
    max_attempts = max_retries + 1
    requests_started = 0

    for attempt in range(max_attempts):
        if _cancel_requested(cancelled):
            telemetry = _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
            )
            raise GeminiCancelledError("Gemini call cancelled", telemetry)

        effective_deadline = operation_deadline
        if deadline_monotonic is not None:
            effective_deadline = min(effective_deadline, float(deadline_monotonic))
        remaining_s = effective_deadline - time.monotonic()
        if remaining_s <= 0:
            telemetry = _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
            )
            raise GeminiDeadlineExceededError("Gemini call deadline exceeded", telemetry)
        request_timeout_s = min(float(timeout_s), remaining_s)

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
        if client is None:
            client = get_client()

        try:
            requests_started += 1
            response = client.models.generate_content(
                model=mdl, contents=user, config=request_config,
            )
        except Exception as error:  # noqa: BLE001
            retryable = _transient_gemini_error(error)
            telemetry = _call_telemetry(
                model=mdl, operation=operation, prompt_version=prompt_version,
                thinking_level=level, started=started,
                retries=max(0, requests_started - 1),
                error=error,
                retryable=retryable,
            )
            if _cancel_requested(cancelled):
                raise GeminiCancelledError("Gemini call cancelled", telemetry) from error
            if time.monotonic() >= effective_deadline:
                raise GeminiDeadlineExceededError(
                    "Gemini call deadline exceeded", telemetry,
                ) from error
            if not retryable or attempt + 1 >= max_attempts:
                raise GeminiTransportError(str(error), telemetry) from error

            wait_s = random.uniform(0.1, 0.35)
            remaining_s = effective_deadline - time.monotonic()
            if remaining_s <= wait_s:
                raise GeminiDeadlineExceededError(
                    "Gemini call deadline exceeded before retry", telemetry,
                ) from error
            time.sleep(wait_s)
            continue

        telemetry = _call_telemetry(
            model=mdl, operation=operation, prompt_version=prompt_version,
            thinking_level=level, started=started,
            retries=max(0, requests_started - 1), response=response,
        )
        if _cancel_requested(cancelled):
            raise GeminiCancelledError("Gemini call cancelled", telemetry)
        if time.monotonic() >= effective_deadline:
            raise GeminiDeadlineExceededError("Gemini call deadline exceeded", telemetry)
        finish_reason = (telemetry.finish_reason or "").upper()
        if finish_reason.endswith("MAX_TOKENS"):
            raise GeminiTruncatedResponseError(
                "Gemini response reached max_output_tokens", telemetry,
            )
        if finish_reason and not finish_reason.endswith("STOP"):
            raise GeminiBlockedResponseError(
                f"Gemini response did not finish normally ({telemetry.finish_reason})",
                telemetry,
            )
        try:
            text = response.text
        except Exception:  # noqa: BLE001 - no usable text is an invalid response
            text = None
        if not text or not str(text).strip():
            raise GeminiEmptyResponseError("empty response from Gemini", telemetry)
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
    """A Gemini video Part from raw mp4 bytes, sent INLINE (use when the request is small,
    < ~20MB — the 8s edge probe). ``media_resolution`` is a ``types.MediaResolution`` enum
    (pass ``types.MediaResolution.MEDIA_RESOLUTION_LOW`` for LOW). Files-API upload +
    video_metadata offsets (Tier 2 / VID3) are intentionally NOT implemented here."""
    if media_resolution is not None:
        return types.Part.from_bytes(data=data, mime_type="video/mp4",
                                     media_resolution=media_resolution)
    return types.Part.from_bytes(data=data, mime_type="video/mp4")


def generate_json_video(system: str, parts: list, schema, media_resolution=None,
                        temperature: float = 0.2, model: Optional[str] = None) -> str:
    """Return a JSON string conforming to ``schema`` from a VIDEO+text prompt.

    ``parts`` is a list of ``types.Part`` (interleaved text + inline video bytes), sent as the
    single user turn. Gemini 2.5 retains the thinking-off compatibility path; Gemini 3
    uses its model-appropriate thinking level without custom sampling.
    Gemini-only (Groq has no video input); the caller invokes this directly, never through
    llm.py/llm_json. Defaults to ``config.VIDEO_JUDGE_MODEL`` (flash-lite)."""
    mdl = model or config.VIDEO_JUDGE_MODEL
    if _is_gemini3_model(mdl):
        return generate_json_v3(
            system,
            parts,
            schema,
            model=mdl,
            thinking_level=_default_gemini3_thinking_level(mdl),
            max_output_tokens=8192,
            timeout_s=120.0,
            deadline_monotonic=None,
            operation="generate_json_video",
            prompt_version="legacy",
            max_retries=1,
            media_resolution=media_resolution,
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
        if media_resolution is not None:
            kwargs["media_resolution"] = media_resolution
        if not thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    def _call():
        try:
            resp = client.models.generate_content(
                model=mdl, contents=parts, config=_make_config(thinking=False)
            )
        except Exception:
            resp = client.models.generate_content(
                model=mdl, contents=parts, config=_make_config(thinking=True)
            )
        return resp.text

    return _retry(_call)


def generate_json_mm(system: str, parts: list, schema, temperature: float = 0.2) -> str:
    """Return a JSON string conforming to ``schema`` from a multimodal prompt.

    ``parts`` is a list of ``types.Part`` (interleaved text + image bytes), sent as the
    single user turn. Gemini 2.5 retains the legacy thinking-budget behavior; Gemini 3
    uses a thinking level and default sampling.
    """
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
