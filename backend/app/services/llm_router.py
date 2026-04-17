"""Central LLM router: Gemini primary, Groq fallback.

Replaces the previous OpenAI-based clients across the backend. Two public
surfaces:

- ``chat_completion(...)`` — text completion with optional JSON mode and
  ``llm_cache`` integration.
- ``transcribe_audio(...)`` — Groq-hosted ``whisper-large-v3`` ASR. Returns
  a list of ``IngestTranscriptCue`` with word-level timestamps when the API
  provides them.

The Gemini key rotation / builder helpers are copied from
``services.topic_cut`` so this module has no runtime dependency on that
file (it would be circular for e.g. reels.py → llm_router → topic_cut).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from ..db import DatabaseIntegrityError, dumps_json, fetch_one, now_iso, upsert

logger = logging.getLogger(__name__)


GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GROQ_AUDIO_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
# Groq currently caps audio uploads at 25 MB.
GROQ_WHISPER_MAX_BYTES = 25 * 1024 * 1024


_gemini_key_offset: int = 0


def _collect_gemini_api_keys() -> list[str]:
    keys: list[str] = []
    primary = os.environ.get("GEMINI_API_KEY") or ""
    if primary:
        keys.append(primary)
    for i in range(2, 20):
        k = os.environ.get(f"GEMINI_API_KEY_{i}") or ""
        if k:
            keys.append(k)
    return keys


def _build_gemini_module(api_key: str | None = None) -> Any | None:
    key = api_key or os.environ.get("GEMINI_API_KEY") or ""
    if not key:
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        logger.debug("google-generativeai not installed; Gemini disabled")
        return None
    try:
        genai.configure(api_key=key)
        return genai
    except Exception:
        logger.exception("could not configure Gemini client")
        return None


def _build_groq_client() -> Any | None:
    api_key = os.environ.get("GROQ_API_KEY") or ""
    if not api_key:
        return None
    try:
        from groq import Groq
    except ImportError:
        logger.debug("groq package not installed; Groq disabled")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        logger.exception("could not build Groq client")
        return None


def gemini_or_groq_available() -> bool:
    return bool(_collect_gemini_api_keys()) or bool(os.environ.get("GROQ_API_KEY") or "")


def _looks_rate_limited(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(marker in s for marker in ("429", "resource_exhausted", "rate", "quota"))


def _read_cache(conn: Any, cache_key: str) -> str | None:
    try:
        row = fetch_one(
            conn,
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        )
    except Exception:
        logger.exception("llm_router: cache read failed for %s", cache_key)
        return None
    if not row:
        return None
    raw = row.get("response_json")
    return raw if isinstance(raw, str) and raw else None


def _write_cache(conn: Any, cache_key: str, content: str) -> None:
    try:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": cache_key,
                "response_json": content,
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
    except DatabaseIntegrityError:
        pass
    except Exception:
        logger.exception("llm_router: cache write failed for %s", cache_key)


def _build_cache_key(namespace: str, model: str, system: str, user: str, *, json_mode: bool) -> str:
    payload = f"{namespace}|{model}|{int(json_mode)}|{system}|{user}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return f"{namespace}:{digest}"


def _gemini_chat(
    *,
    genai_module: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
    max_output_tokens: int | None,
) -> str | None:
    global _gemini_key_offset

    all_keys = _collect_gemini_api_keys() or [""]
    last_exc: Exception | None = None

    for attempt in range(len(all_keys)):
        idx = (_gemini_key_offset + attempt) % len(all_keys)
        key = all_keys[idx]
        if key:
            try:
                genai_module.configure(api_key=key)
            except Exception:
                continue

        try:
            generation_config_kwargs: dict[str, Any] = {"temperature": float(temperature)}
            if json_mode:
                generation_config_kwargs["response_mime_type"] = "application/json"
            if max_output_tokens is not None:
                generation_config_kwargs["max_output_tokens"] = int(max_output_tokens)
            model_obj = genai_module.GenerativeModel(
                model_name=model,
                system_instruction=system,
                generation_config=genai_module.GenerationConfig(**generation_config_kwargs),
            )
            response = model_obj.generate_content(user)
            text = (response.text or "").strip()
            if not text:
                return None
            _gemini_key_offset = idx
            return text
        except Exception as exc:
            last_exc = exc
            if _looks_rate_limited(exc) and len(all_keys) > 1:
                logger.info(
                    "llm_router.gemini: key %d/%d rate-limited; rotating",
                    idx + 1, len(all_keys),
                )
                _gemini_key_offset = (idx + 1) % len(all_keys)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return None


def _groq_chat(
    *,
    groq_client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
    max_tokens: int | None,
) -> str | None:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    response = groq_client.chat.completions.create(**kwargs)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    content = getattr(choices[0].message, "content", None) or ""
    return content.strip() or None


def chat_completion(
    *,
    conn: Any | None = None,
    cache_key: str | None = None,
    cache_namespace: str | None = None,
    system: str,
    user: str,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int | None = None,
    gemini_model: str = GEMINI_DEFAULT_MODEL,
    groq_model: str = GROQ_DEFAULT_MODEL,
) -> str | None:
    """Run a chat completion: Gemini first, Groq fallback.

    Returns the raw model text (already stripped) or ``None`` if neither
    provider is available or both failed.

    Caching:
      * If ``cache_key`` is supplied, that exact string is used.
      * Else if ``cache_namespace`` is supplied along with a ``conn``, a cache
        key is derived from the namespace + model + prompt hash.
      * Caching is disabled if neither is supplied or if ``conn`` is None.
    """
    effective_cache_key: str | None = None
    if conn is not None:
        if cache_key:
            effective_cache_key = cache_key
        elif cache_namespace:
            effective_cache_key = _build_cache_key(
                cache_namespace, gemini_model, system, user, json_mode=json_mode
            )
        if effective_cache_key:
            cached = _read_cache(conn, effective_cache_key)
            if cached is not None:
                return cached

    gemini = _build_gemini_module()
    if gemini is not None:
        try:
            out = _gemini_chat(
                genai_module=gemini,
                model=gemini_model,
                system=system,
                user=user,
                temperature=temperature,
                json_mode=json_mode,
                max_output_tokens=max_tokens,
            )
            if out:
                if conn is not None and effective_cache_key:
                    _write_cache(conn, effective_cache_key, out)
                return out
        except Exception:
            logger.warning("llm_router.chat_completion: Gemini failed, falling back to Groq", exc_info=True)

    groq = _build_groq_client()
    if groq is not None:
        try:
            out = _groq_chat(
                groq_client=groq,
                model=groq_model,
                system=system,
                user=user,
                temperature=temperature,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
            if out:
                if conn is not None and effective_cache_key:
                    _write_cache(conn, effective_cache_key, out)
                return out
        except Exception:
            logger.warning("llm_router.chat_completion: Groq failed", exc_info=True)

    return None


def transcribe_audio(
    audio_path: Path,
    *,
    language: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any] | None:
    """Transcribe a local audio file via Groq's hosted Whisper endpoint.

    Returns a dict shaped like OpenAI's ``verbose_json`` response:
    ``{"segments": [...], "words": [...], "text": "..."}``. Returns ``None``
    if no ``GROQ_API_KEY`` is configured (caller should fall back).

    Raises ``RuntimeError`` on HTTP/transport failures so the caller can
    wrap with its domain-specific error (e.g. ``TranscriptionError``).
    """
    api_key = os.environ.get("GROQ_API_KEY") or ""
    if not api_key:
        return None

    try:
        size = audio_path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"audio file is missing: {exc}") from exc
    if size == 0:
        raise RuntimeError("audio file is empty")
    if size > GROQ_WHISPER_MAX_BYTES:
        raise RuntimeError(
            f"audio file too large for Groq Whisper (size={size}, max={GROQ_WHISPER_MAX_BYTES})"
        )

    data: dict[str, str] = {
        "model": GROQ_WHISPER_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }
    if language:
        data["language"] = language

    # Groq accepts repeated "timestamp_granularities[]" form fields.
    # httpx handles repeats via a list of tuples in the data payload.
    form_items = [
        ("model", GROQ_WHISPER_MODEL),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
    ]
    if language:
        form_items.append(("language", language))

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with open(audio_path, "rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/wav")}
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    GROQ_AUDIO_ENDPOINT,
                    headers=headers,
                    data=form_items,
                    files=files,
                )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Groq Whisper request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Groq Whisper returned {resp.status_code}: {resp.text[:400]}"
        )
    try:
        payload = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Groq Whisper returned non-JSON response: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Groq Whisper response was not a JSON object")
    return payload
