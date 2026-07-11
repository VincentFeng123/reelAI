"""Gemini (Google AI Studio) wrapper for structured JSON selection output.

Uses the google-genai SDK's `models.generate_content` with `response_mime_type=
application/json` + a Pydantic `response_schema`, which enforces the output shape.
Includes a simple retry on rate-limit / transient errors.
"""
from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from . import config
from ..cancellation import raise_if_cancelled, run_cancellable, sleep_with_probe
from ..errors import (
    CancellationError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTransientError,
)
from ..provider_runtime import GenerationContext, MAX_PROVIDER_RETRIES, bounded_retry_after

_client: Optional[genai.Client] = None
_lock = threading.Lock()


def _create_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise ProviderConfigurationError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.",
            provider="gemini",
            operation="segmentation",
        )
    return genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=120_000),  # ms; don't hang forever
    )


def get_client() -> genai.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _create_client()
    return _client


def _error_code(e: Exception) -> Optional[int]:
    code = getattr(e, "code", None)
    if code is None:
        code = getattr(e, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(e: Exception) -> bool:
    # Prefer the structured google-genai APIError code: 403/404 (no access /
    # bad model name) must fail fast, not burn 6 backoff rounds.
    code = _error_code(e)
    if code is not None:
        return code == 429 or 500 <= code <= 599
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return True
    s = str(e).lower()
    return (
        "429" in s or "resource_exhausted" in s or "rate limit" in s or "rate_limit" in s
        or "503" in s or "500" in s or "unavailable" in s or "overloaded" in s
        or "handler is closed" in s or "client has been closed" in s
    )


def is_model_unavailable(e: Exception) -> bool:
    """A non-transient model-access failure (bad name / no access): callers may
    fall back to the default model instead of failing the whole video."""
    if isinstance(e, ModelUnavailableError):
        return True
    code = _error_code(e)
    if code == 404:
        return True
    s = str(e).lower()
    return (
        "not_found" in s or "not found" in s
    )


def _exception_headers(exc: Exception):
    headers = getattr(exc, "headers", None)
    if headers is not None:
        return headers
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None)


def _typed_failure(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    code = _error_code(exc)
    detail = str(exc)[:500]
    kwargs = dict(
        provider="gemini",
        operation="segmentation",
        status_code=code,
        detail=detail or None,
    )
    if code in (401, 403):
        return ProviderAuthenticationError("Gemini authentication failed.", **kwargs)
    if code == 402:
        return ProviderQuotaError("Gemini quota is exhausted.", **kwargs)
    if code == 404:
        return ModelUnavailableError("The configured Gemini model is unavailable.", **kwargs)
    if code == 429:
        return ProviderRateLimitError(
            "Gemini is rate limited.",
            retry_after_sec=bounded_retry_after(_exception_headers(exc)),
            **kwargs,
        )
    lowered = detail.lower()
    if code is None and (
        "429" in lowered
        or "resource_exhausted" in lowered
        or "rate limit" in lowered
        or "rate_limit" in lowered
    ):
        return ProviderRateLimitError(
            "Gemini is rate limited.",
            retry_after_sec=bounded_retry_after(_exception_headers(exc)),
            **kwargs,
        )
    if code is not None and 500 <= code <= 599:
        return ProviderTransientError("Gemini is temporarily unavailable.", **kwargs)
    if code is None and _is_retryable(exc):
        return ProviderTransientError("Could not reach Gemini.", **kwargs)
    return ProviderRequestError("Gemini rejected the segmentation request.", **kwargs)


@dataclass(frozen=True)
class GeminiResponse:
    text: str
    model_used: str
    quality_degraded: bool
    usage: object | None = None


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


async def _retry_async(
    call,
    should_cancel: Callable[[], bool] | None = None,
    *,
    context: GenerationContext | None = None,
    model_used: str,
    quality_degraded: bool,
) -> GeminiResponse:
    """Async retry loop whose request and backoff both observe cancellation."""
    last: Optional[Exception] = None
    for retry_index in range(MAX_PROVIDER_RETRIES + 1):
        raise_if_cancelled(should_cancel)
        attempt = retry_index + 1
        if context is not None:
            context.reserve("segmentation")
        try:
            response = await call()
            text = str(getattr(response, "text", "") or "")
            if not text:
                raise ProviderRequestError(
                    "Gemini returned an empty segmentation response.",
                    provider="gemini",
                    operation="segmentation",
                )
            actual_model = str(getattr(response, "model_version", "") or model_used)
            usage = getattr(response, "usage_metadata", None)
            if context is not None:
                context.record_gemini(
                    attempt=attempt,
                    model_used=actual_model,
                    quality_degraded=quality_degraded,
                    usage=usage,
                )
            return GeminiResponse(text, actual_model, quality_degraded, usage)
        except asyncio.CancelledError:
            raise
        except CancellationError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            typed = _typed_failure(exc)
            if context is not None:
                context.record_gemini(
                    attempt=attempt,
                    model_used=model_used,
                    quality_degraded=quality_degraded,
                    status_code=typed.status_code,
                    error_code=typed.code,
                )
            if not typed.retryable or retry_index >= MAX_PROVIDER_RETRIES:
                raise typed from exc
        wait = bounded_retry_after(_exception_headers(last))
        if wait is None:
            wait = min(config.BACKOFF_CAP, config.BACKOFF_BASE * 2 ** retry_index)
        await sleep_with_probe(wait, should_cancel)
    raise ProviderTransientError(
        f"Gemini call failed: {last}", provider="gemini", operation="segmentation"
    )


def generate_json_result(system: str, user: str, schema, temperature: float = 0.2,
                         model: Optional[str] = None, max_output_tokens: int = 8192,
                         should_cancel: Callable[[], bool] | None = None,
                         *, context: GenerationContext | None = None,
                         quality_degraded: bool = False) -> GeminiResponse:
    """Return a JSON string conforming to `schema` (a Pydantic model class).

    ``max_output_tokens`` bounds the response; raise it for calls whose JSON is large
    OR that use a thinking model (thinking tokens share this budget, so an 8192 cap can
    truncate the JSON on a preview/pro model that ignores thinking_budget=0)."""
    mdl = model or config.GEMINI_MODEL

    def _make_config():
        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    async def _call():
        raise_if_cancelled(should_cancel)
        client = _create_client()
        try:
            return await client.aio.models.generate_content(
                model=mdl, contents=user, config=_make_config()
            )
        finally:
            try:
                await client.aio.aclose()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass

    return run_cancellable(
        lambda: _retry_async(
            _call,
            should_cancel,
            context=context,
            model_used=mdl,
            quality_degraded=quality_degraded,
        ),
        should_cancel,
    )


def generate_json(system: str, user: str, schema, temperature: float = 0.2,
                  model: Optional[str] = None, max_output_tokens: int = 8192,
                  should_cancel: Callable[[], bool] | None = None,
                  *, context: GenerationContext | None = None,
                  quality_degraded: bool = False) -> str:
    return generate_json_result(
        system,
        user,
        schema,
        temperature=temperature,
        model=model,
        max_output_tokens=max_output_tokens,
        should_cancel=should_cancel,
        context=context,
        quality_degraded=quality_degraded,
    ).text


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
    single user turn — same thinking-off-then-on + backoff machinery as ``generate_json_mm``.
    Gemini-only; the caller invokes this directly, never through
    llm.py/llm_json. Defaults to ``config.VIDEO_JUDGE_MODEL`` (flash-lite)."""
    client = get_client()
    mdl = model or config.VIDEO_JUDGE_MODEL

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
    single user turn. Same thinking-off-then-on + backoff dance as ``generate_json``.
    """
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
