"""Central text-LLM router: Gemini primary with a local Ollama fallback.

The public ``chat_completion(...)`` surface supports optional structured JSON
and ``llm_cache`` integration. Groq and Cerebras remain deliberately disabled.

The Gemini key rotation / builder helpers are copied from
``services.topic_cut`` so this module has no runtime dependency on that
file (it would be circular for e.g. reels.py → llm_router → topic_cut).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from ..db import DatabaseIntegrityError, dumps_json, fetch_one, now_iso, upsert
from ..clip_engine.cancellation import raise_if_cancelled, run_cancellable
from ..clip_engine.errors import CancellationError

logger = logging.getLogger(__name__)


GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
# llama3.1-8b is the default Cerebras free-tier model available to every
# account (the other universally-available one is gpt-oss-120b). The
# earlier default `llama-3.3-70b` returned 404 for accounts that weren't
# explicitly provisioned for it. Override via CEREBRAS_MODEL env var.
CEREBRAS_DEFAULT_MODEL = os.environ.get("CEREBRAS_MODEL", "llama3.1-8b").strip() or "llama3.1-8b"
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip() or "qwen2.5:7b-instruct"


def _bounded_timeout(name: str, default: float) -> float:
    try:
        configured = float(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        configured = default
    return max(1.0, min(300.0, configured))


OLLAMA_DEFAULT_TIMEOUT = _bounded_timeout("OLLAMA_TIMEOUT", 180.0)
GEMINI_DEFAULT_TIMEOUT = _bounded_timeout("GEMINI_TIMEOUT", 30.0)


_gemini_key_offset: int = 0


class TextLLMUnavailableError(RuntimeError):
    """No configured text provider completed the request."""

    code = "text_llm_unavailable"
    retryable = True

    def __init__(self, message: str = "No text model is reachable right now.") -> None:
        super().__init__(message)
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


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
    """Return the ``google.genai`` module when a key is available.

    The new SDK builds a fresh ``Client`` per call (see ``_gemini_chat``),
    so this helper only checks whether the module is importable and a key
    is present — it does not construct a client here.
    """
    key = api_key or next(iter(_collect_gemini_api_keys()), "")
    if not key:
        return None
    try:
        from google import genai
    except ImportError:
        logger.debug("google-genai not installed; Gemini disabled")
        return None
    return genai


def _build_ollama_client() -> dict[str, Any] | None:
    """Return an Ollama config dict when ``OLLAMA_BASE_URL`` is set.

    Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint,
    so the "client" is just a base URL + model name; we call it with httpx.
    Gating on an explicit env var means production (where localhost:11434
    is not running) never accidentally enables this path.
    """
    base_url = (os.environ.get("OLLAMA_BASE_URL") or "").strip()
    if not base_url:
        return None
    return {"base_url": base_url.rstrip("/"), "model": OLLAMA_DEFAULT_MODEL}


def _build_groq_client() -> Any | None:
    # PR 4: API blackout — Groq disabled; router falls back to Ollama/heuristic.
    return None
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


def _build_cerebras_client() -> Any | None:
    # PR 4: API blackout — Cerebras disabled; router falls back to Ollama/heuristic.
    return None
    api_key = os.environ.get("CEREBRAS_API_KEY") or ""
    if not api_key:
        return None
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError:
        logger.debug("cerebras-cloud-sdk not installed; Cerebras disabled")
        return None
    try:
        # timeout=10s + max_retries=0 fast-fails instead of hanging for
        # minutes when the API is overloaded or the key is bad — lets the
        # chain fall through to the heuristic picker quickly.
        return Cerebras(api_key=api_key, timeout=10.0, max_retries=0)
    except Exception:
        logger.exception("could not build Cerebras client")
        return None


def provider_availability(
    *,
    gemini_api_key_override: str | None = None,
) -> dict[str, bool]:
    """Report providers the router can actually invoke in this process."""
    return {
        "ollama": _build_ollama_client() is not None,
        "gemini": _build_gemini_module(api_key=gemini_api_key_override) is not None,
        "groq": _build_groq_client() is not None,
        "cerebras": _build_cerebras_client() is not None,
    }


def text_llm_status(
    *,
    gemini_model: str = GEMINI_DEFAULT_MODEL,
    gemini_api_key_override: str | None = None,
) -> dict[str, Any]:
    providers = provider_availability(
        gemini_api_key_override=gemini_api_key_override,
    )
    if providers["gemini"]:
        provider = "gemini"
        model = gemini_model
    elif providers["ollama"]:
        provider = "ollama"
        model = OLLAMA_DEFAULT_MODEL
    else:
        provider = None
        model = None
    return {
        "available": provider is not None,
        "provider": provider,
        "model": model,
        "providers": providers,
    }


def gemini_or_groq_available(
    *,
    gemini_api_key_override: str | None = None,
) -> bool:
    """Compatibility surface for callers that only need text-LLM readiness."""
    return bool(
        text_llm_status(
            gemini_api_key_override=gemini_api_key_override,
        )["available"]
    )


def _log_provider_outcome(
    provider: str,
    outcome: str,
    error: BaseException | None = None,
) -> None:
    """Log operational outcomes without prompts, credentials, or model output."""
    fields: dict[str, Any] = {"provider": provider, "outcome": outcome}
    if error is not None:
        fields["error_type"] = type(error).__name__
    logger.info("text_llm_outcome %s", fields)


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


def _build_cache_key(
    namespace: str,
    model: str,
    system: str,
    user: str,
    *,
    json_mode: bool,
    response_schema: type[BaseModel] | None = None,
) -> str:
    schema_token = ""
    if response_schema is not None:
        schema_json = json.dumps(
            response_schema.model_json_schema(),
            sort_keys=True,
            separators=(",", ":"),
        )
        schema_token = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()[:16]
    if schema_token:
        payload = f"{namespace}|{model}|{int(json_mode)}|{schema_token}|{system}|{user}"
    else:
        payload = f"{namespace}|{model}|{int(json_mode)}|{system}|{user}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return f"{namespace}:{digest}"


def _validated_output(
    content: str,
    response_schema: type[BaseModel] | None,
) -> str:
    clean = str(content or "").strip()
    if response_schema is None:
        return clean
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    value = response_schema.model_validate_json(clean)
    return value.model_dump_json(by_alias=True)


def _ollama_chat(
    *,
    ollama_client: dict[str, Any],
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
    max_tokens: int | None,
    should_cancel: Callable[[], bool] | None = None,
) -> str | None:
    """Call an Ollama server via its OpenAI-compatible chat endpoint."""
    payload: dict[str, Any] = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    async def _request() -> str | None:
        raise_if_cancelled(should_cancel)
        url = f"{ollama_client['base_url']}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=OLLAMA_DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        content = (choices[0].get("message") or {}).get("content") or ""
        return content.strip() or None

    return run_cancellable(_request, should_cancel)


def _gemini_chat(
    *,
    genai_module: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
    max_output_tokens: int | None,
    response_schema: type[BaseModel] | None = None,
    api_key_override: str | None = None,
    key_attempt_limit: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str | None:
    global _gemini_key_offset

    if api_key_override:
        # Dedicated key path: no rotation, no fallback to the main pool.
        all_keys = [api_key_override]
    else:
        all_keys = _collect_gemini_api_keys() or [""]
    last_exc: Exception | None = None

    from google.genai import types as genai_types

    key_attempts = len(all_keys)
    if key_attempt_limit is not None:
        key_attempts = min(key_attempts, max(1, int(key_attempt_limit)))

    for attempt in range(key_attempts):
        raise_if_cancelled(should_cancel)
        idx = (_gemini_key_offset + attempt) % len(all_keys) if not api_key_override else 0
        key = all_keys[idx]
        if not key:
            continue
        try:
            client_kwargs: dict[str, Any] = {"api_key": key}
            if key_attempt_limit is not None:
                client_kwargs["http_options"] = genai_types.HttpOptions(
                    timeout=max(10_000, int(GEMINI_DEFAULT_TIMEOUT * 1_000)),
                    retry_options=genai_types.HttpRetryOptions(attempts=1),
                )
            client = genai_module.Client(**client_kwargs)
        except Exception:
            continue

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": float(temperature),
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = response_schema.model_json_schema()
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_output_tokens)

        try:
            async def _request() -> Any:
                aio_client = client.aio
                try:
                    return await asyncio.wait_for(
                        aio_client.models.generate_content(
                            model=model,
                            contents=user,
                            config=genai_types.GenerateContentConfig(**config_kwargs),
                        ),
                        timeout=GEMINI_DEFAULT_TIMEOUT,
                    )
                finally:
                    async_close = getattr(aio_client, "aclose", None)
                    if callable(async_close):
                        close_result = async_close()
                        if inspect.isawaitable(close_result):
                            await close_result
                    sync_close = getattr(client, "close", None)
                    if callable(sync_close):
                        sync_close()

            response = run_cancellable(
                _request,
                should_cancel,
            )
            text = (response.text or "").strip()
            if not text:
                return None
            _gemini_key_offset = idx
            return text
        except Exception as exc:
            last_exc = exc
            if _looks_rate_limited(exc) and len(all_keys) > 1:
                logger.info(
                    "llm_router.gemini: credential slot %d/%d rate-limited; rotating",
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


def _cerebras_chat(
    *,
    cerebras_client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
    max_tokens: int | None,
) -> str | None:
    """Cerebras Llama 3.3 70B call — OpenAI-compatible, same shape as Groq."""
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
        kwargs["max_completion_tokens"] = int(max_tokens)
    response = cerebras_client.chat.completions.create(**kwargs)
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
    response_schema: type[BaseModel] | None = None,
    max_tokens: int | None = None,
    gemini_model: str = GEMINI_DEFAULT_MODEL,
    groq_model: str = GROQ_DEFAULT_MODEL,
    cerebras_model: str = CEREBRAS_DEFAULT_MODEL,
    gemini_api_key_override: str | None = None,
    gemini_key_attempt_limit: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str | None:
    """Run a chat completion: Gemini first, Groq fallback.

    Returns the raw model text (already stripped) or ``None`` if neither
    provider is available or both failed.

    Caching:
      * If ``cache_key`` is supplied, that exact string is used.
      * Else if ``cache_namespace`` is supplied along with a ``conn``, a cache
        key is derived from the namespace + model + prompt hash.
      * Caching is disabled if neither is supplied or if ``conn`` is None.

    ``gemini_api_key_override`` forces a specific key for the Gemini call
    (used by the chat endpoint so it runs on a dedicated backup key without
    consuming the primary rotation pool's quota).

    ``response_schema`` enables Gemini structured output and validates both
    cached and provider responses before returning canonical JSON text.
    """
    raise_if_cancelled(should_cancel)
    effective_json_mode = bool(json_mode or response_schema is not None)
    effective_cache_key: str | None = None
    if conn is not None:
        if cache_key:
            effective_cache_key = cache_key
        elif cache_namespace:
            effective_cache_key = _build_cache_key(
                cache_namespace,
                gemini_model,
                system,
                user,
                json_mode=effective_json_mode,
                response_schema=response_schema,
            )
        if effective_cache_key:
            cached = _read_cache(conn, effective_cache_key)
            if cached is not None:
                try:
                    validated = _validated_output(cached, response_schema)
                except (ValueError, TypeError) as exc:
                    _log_provider_outcome("cache", "invalid", exc)
                else:
                    _log_provider_outcome("cache", "hit")
                    return validated

    gemini = _build_gemini_module(api_key=gemini_api_key_override)
    if gemini is not None:
        try:
            out = _gemini_chat(
                genai_module=gemini,
                model=gemini_model,
                system=system,
                user=user,
                temperature=temperature,
                json_mode=effective_json_mode,
                response_schema=response_schema,
                max_output_tokens=max_tokens,
                api_key_override=gemini_api_key_override,
                key_attempt_limit=gemini_key_attempt_limit,
                should_cancel=should_cancel,
            )
            if out:
                out = _validated_output(out, response_schema)
                raise_if_cancelled(should_cancel)
                if conn is not None and effective_cache_key:
                    _write_cache(conn, effective_cache_key, out)
                _log_provider_outcome("gemini", "success")
                return out
            _log_provider_outcome("gemini", "empty")
        except CancellationError:
            raise
        except Exception as exc:
            _log_provider_outcome("gemini", "failed", exc)

    ollama = _build_ollama_client()
    if ollama is not None:
        try:
            out = _ollama_chat(
                ollama_client=ollama,
                model=ollama["model"],
                system=system,
                user=user,
                temperature=temperature,
                json_mode=effective_json_mode,
                max_tokens=max_tokens,
                should_cancel=should_cancel,
            )
            if out:
                out = _validated_output(out, response_schema)
                raise_if_cancelled(should_cancel)
                if conn is not None and effective_cache_key:
                    _write_cache(conn, effective_cache_key, out)
                _log_provider_outcome("ollama", "success")
                return out
            _log_provider_outcome("ollama", "empty")
        except CancellationError:
            raise
        except Exception as exc:
            _log_provider_outcome("ollama", "failed", exc)

    groq = _build_groq_client()
    if groq is not None:
        try:
            out = _groq_chat(
                groq_client=groq,
                model=groq_model,
                system=system,
                user=user,
                temperature=temperature,
                json_mode=effective_json_mode,
                max_tokens=max_tokens,
            )
            if out:
                out = _validated_output(out, response_schema)
                if conn is not None and effective_cache_key:
                    _write_cache(conn, effective_cache_key, out)
                _log_provider_outcome("groq", "success")
                return out
            _log_provider_outcome("groq", "empty")
        except Exception as exc:
            _log_provider_outcome("groq", "failed", exc)

    cerebras = _build_cerebras_client()
    if cerebras is not None:
        try:
            out = _cerebras_chat(
                cerebras_client=cerebras,
                model=cerebras_model,
                system=system,
                user=user,
                temperature=temperature,
                json_mode=effective_json_mode,
                max_tokens=max_tokens,
            )
            if out:
                out = _validated_output(out, response_schema)
                if conn is not None and effective_cache_key:
                    _write_cache(conn, effective_cache_key, out)
                _log_provider_outcome("cerebras", "success")
                return out
            _log_provider_outcome("cerebras", "empty")
        except Exception as exc:
            _log_provider_outcome("cerebras", "failed", exc)

    return None
