"""Gemini (Google AI Studio) wrapper for structured JSON selection output.

Uses the google-genai SDK's `models.generate_content` with `response_mime_type=
application/json` + a Pydantic `response_schema`, which enforces the output shape.
Includes a simple retry on rate-limit / transient errors.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Optional

from google import genai
from google.genai import types

from . import config

_client: Optional[genai.Client] = None
_lock = threading.Lock()


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
        return code in (429, 500, 503)
    s = str(e).lower()
    return (
        "429" in s or "resource_exhausted" in s or "rate" in s
        or "503" in s or "500" in s or "unavailable" in s or "overloaded" in s
    )


def is_model_unavailable(e: Exception) -> bool:
    """A non-transient model-access failure (bad name / no access): callers may
    fall back to the default model instead of failing the whole video."""
    code = _error_code(e)
    if code in (403, 404):
        return True
    s = str(e).lower()
    return (
        "not_found" in s or "not found" in s
        or "permission_denied" in s or "permission denied" in s
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


def generate_json(system: str, user: str, schema, temperature: float = 0.2,
                  model: Optional[str] = None, max_output_tokens: int = 8192) -> str:
    """Return a JSON string conforming to `schema` (a Pydantic model class).

    ``max_output_tokens`` bounds the response; raise it for calls whose JSON is large
    OR that use a thinking model (thinking tokens share this budget, so an 8192 cap can
    truncate the JSON on a preview/pro model that ignores thinking_budget=0)."""
    client = get_client()
    mdl = model or config.GEMINI_MODEL

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
