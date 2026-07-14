"""Query expansion for the existing planner and the opt-in practice-fast path.

Topic correction, aliases, and semantic expansion are owned by the persisted
TopicExpansionService. The existing ``expand_query`` API retains its stable templates.
Production retrieval uses ``expand_query_practice_fast`` for one cached Flash call and
falls back only to the user's literal request when the provider is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel

from . import config
from .cancellation import raise_if_cancelled, run_cancellable
from .errors import CancellationError
from .segment_cache import SEGMENT_CACHE_TTL_SEC
from .singleflight import singleflight
from ..db import dumps_json, fetch_one, get_conn, now_iso, upsert

if TYPE_CHECKING:
    from .provider_runtime import GenerationContext

logger = logging.getLogger(__name__)

PRACTICE_FAST_EXPAND_MODEL = "gemini-3.1-flash-lite"
PRACTICE_FAST_EXPAND_TIMEOUT_MS = 8_000
PRACTICE_FAST_EXPAND_OUTPUT_TOKENS = 1_024
PRACTICE_FAST_EXPAND_CACHE_VERSION = 3
# An expansion can be nearly one segment-cache lifetime old when it discovers a
# newly analyzed source. Keeping it for two lifetimes guarantees that source's
# subsequent valid segment-cache lifetime never triggers another expansion call.
PRACTICE_FAST_EXPAND_CACHE_TTL_SEC = 2 * SEGMENT_CACHE_TTL_SEC


class _PracticeFastExpansion(BaseModel):
    corrected: str
    queries: list[str]


_PRACTICE_FAST_SYSTEM = """You expand a user's search topic into a diverse set of
YouTube search queries that maximize topical coverage.

Do this:
1. Spellcheck and correct the user's input.
2. Infer the most likely intent or sense.
3. Produce up to N concise queries that a person would actually search on YouTube.
4. Preserve the user's named subject, task, and requested relationship. Cover close synonyms,
   genuinely related informational facets, useful prerequisites, and educational sources, but
   never redirect the search into a merely adjacent field.

Return only the requested JSON object. Put the corrected topic first in queries."""


def literal_fallback(topic: str, n: int) -> dict:
    literal = " ".join(str(topic or "").split())
    return {
        "corrected": literal,
        "queries": [literal] if literal and int(n) > 0 else [],
        "provider_used": "literal_fallback",
    }


def _expansion_cache_key(topic: str, n: int, level: str | None) -> str:
    contract = {
        "version": PRACTICE_FAST_EXPAND_CACHE_VERSION,
        "topic": topic,
        "count": int(n),
        "level": " ".join(str(level or "").split()),
        "model": PRACTICE_FAST_EXPAND_MODEL,
        "prompt_sha256": hashlib.sha256(_PRACTICE_FAST_SYSTEM.encode("utf-8")).hexdigest(),
        "schema": _PracticeFastExpansion.model_json_schema(),
    }
    encoded = json.dumps(
        contract,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        f"practice-fast-expansion:v{PRACTICE_FAST_EXPAND_CACHE_VERSION}:"
        f"{hashlib.sha256(encoded).hexdigest()}"
    )


def _cache_age_seconds(created_at: object) -> float:
    try:
        parsed = datetime.fromisoformat(str(created_at or "").replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(
        0.0,
        (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds(),
    )


def _read_cached_expansion(cache_key: str, count: int) -> dict | None:
    try:
        with get_conn() as conn:
            row = fetch_one(
                conn,
                "SELECT response_json, created_at FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
    except Exception as exc:
        logger.debug("Practice expansion cache read unavailable: %s", exc)
        return None
    if not row or _cache_age_seconds(row.get("created_at")) >= PRACTICE_FAST_EXPAND_CACHE_TTL_SEC:
        return None
    try:
        payload = json.loads(str(row.get("response_json") or "{}"))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != PRACTICE_FAST_EXPAND_CACHE_VERSION:
        return None
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list) or not all(isinstance(query, str) for query in raw_queries):
        return None
    corrected = " ".join(str(payload.get("corrected") or "").split())
    queries = _normalize(raw_queries, count)
    if not corrected or not queries or _key(corrected) != _key(queries[0]):
        return None
    return {
        "corrected": corrected,
        "queries": queries,
        "provider_used": "gemini",
    }


def _write_cached_expansion(cache_key: str, result: dict) -> None:
    try:
        with get_conn(transactional=True) as conn:
            upsert(
                conn,
                "llm_cache",
                {
                    "cache_key": cache_key,
                    "response_json": dumps_json(
                        {
                            "version": PRACTICE_FAST_EXPAND_CACHE_VERSION,
                            "corrected": result["corrected"],
                            "queries": result["queries"],
                        }
                    ),
                    "created_at": now_iso(),
                },
                pk="cache_key",
            )
    except Exception as exc:
        logger.debug("Practice expansion cache write unavailable: %s", exc)


def _key(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _normalize(queries: list[str], n: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = " ".join(str(query or "").split())
        key = _key(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max(0, int(n)):
            break
    return result


def deterministic_expand(topic: str, n: int, *, level: str | None = None) -> dict:
    topic = " ".join(str(topic or "").split())
    level_key = _key(level)
    if level_key == "beginner":
        variants = [
            topic,
            f"introduction to {topic}",
            f"{topic} basics",
            f"{topic} explained",
            f"{topic} for beginners",
            f"{topic} tutorial",
            f"{topic} lecture",
        ]
    elif level_key == "advanced":
        variants = [
            topic,
            f"advanced {topic}",
            f"graduate {topic} lecture",
            f"{topic} deep dive",
            f"{topic} seminar",
            f"{topic} explained",
            f"{topic} lecture",
        ]
    else:
        variants = [
            topic,
            f"{topic} explained",
            f"{topic} lecture",
            f"{topic} tutorial",
            f"how {topic} works",
            f"{topic} course",
            f"{topic} fundamentals",
        ]
    return {
        "corrected": topic,
        "queries": _normalize(variants, n),
        "provider_used": "deterministic",
    }


def free_expand(topic: str, n: int) -> dict:
    """Compatibility alias for the old keyless expansion entry point."""
    return deterministic_expand(topic, n)


async def _practice_fast_gemini_raw_async(
    topic: str,
    n: int,
    *,
    model: str,
    level: str | None,
    should_cancel: Callable[[], bool] | None,
    context: "GenerationContext | None" = None,
) -> str:
    """Make the path's single provider call. There is intentionally no model fallback."""
    raise_if_cancelled(should_cancel)
    key = config.require_gemini_key()
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=PRACTICE_FAST_EXPAND_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    level_text = str(level or "").strip()
    level_line = f"\nViewer level: {level_text}" if level_text else ""
    user = (
        f"User topic: {topic!r}\nN = {max(1, int(n))}{level_line}\n"
        "Return corrected and queries as JSON."
    )
    reservation: dict[str, object] = {}
    try:
        if context is not None:
            reserved = context.reserve_gemini_call(
                operation="expansion",
                model=model,
                prompt_text=f"{_PRACTICE_FAST_SYSTEM}\n\n{user}",
                max_output_tokens=PRACTICE_FAST_EXPAND_OUTPUT_TOKENS,
            )
            if isinstance(reserved, dict):
                reservation = dict(reserved)
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=_PRACTICE_FAST_SYSTEM,
                    response_mime_type="application/json",
                    response_json_schema=_PracticeFastExpansion.model_json_schema(),
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    max_output_tokens=PRACTICE_FAST_EXPAND_OUTPUT_TOKENS,
                ),
            )
        except Exception as exc:
            if context is not None:
                context.record_gemini(
                    operation="expansion",
                    attempt=1,
                    model_used=model,
                    quality_degraded=False,
                    usage={**reservation, "dispatched": True},
                    status_code=None,
                    error_code=f"dispatch_failed:{type(exc).__name__}",
                )
            raise
        raise_if_cancelled(should_cancel)
        if context is not None:
            raw_usage = getattr(response, "usage_metadata", None)

            def usage_field(name: str) -> object:
                if isinstance(raw_usage, dict):
                    return raw_usage.get(name, 0)
                return getattr(raw_usage, name, 0)

            context.record_gemini(
                operation="expansion",
                attempt=1,
                model_used=str(getattr(response, "model_version", "") or model),
                quality_degraded=model != PRACTICE_FAST_EXPAND_MODEL,
                usage={
                    "prompt_token_count": usage_field("prompt_token_count"),
                    "candidates_token_count": usage_field("candidates_token_count"),
                    "thoughts_token_count": usage_field("thoughts_token_count"),
                    "total_token_count": usage_field("total_token_count"),
                    **reservation,
                    "dispatched": True,
                },
            )
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise ValueError("Gemini returned an empty query expansion")
        return text
    finally:
        aio_client = getattr(client, "aio", None)
        async_close = getattr(aio_client, "aclose", None)
        if callable(async_close):
            try:
                await async_close()
            except Exception:
                pass
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _practice_fast_gemini_raw(
    topic: str,
    n: int,
    *,
    model: str = PRACTICE_FAST_EXPAND_MODEL,
    level: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
    context: "GenerationContext | None" = None,
) -> str:
    return run_cancellable(
        lambda: _practice_fast_gemini_raw_async(
            topic,
            n,
            model=model,
            level=level,
            should_cancel=should_cancel,
            context=context,
        ),
        should_cancel,
    )


def expand_query_practice_fast(
    topic: str,
    n: int,
    level: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    context: "GenerationContext | None" = None,
) -> dict:
    """Return cached Flash search terms, failing safely to the literal request.

    Expansion deliberately does not reserve the Supadata search budget tracked by
    ``context``. The context enables durable cache reuse and usage accounting.
    """
    topic = " ".join(str(topic or "").split())
    count = max(0, int(n))
    if not topic or count == 0:
        return literal_fallback(topic, count)
    raise_if_cancelled(should_cancel)
    cache_key = _expansion_cache_key(topic, count, level) if context is not None else ""
    cached = _read_cached_expansion(cache_key, count) if cache_key else None
    if cached is not None:
        context.increment_counter("expansion_cache_hits")
        context.record_cache_hit(
            provider="gemini",
            operation="expansion",
            metadata={"cache_key": cache_key},
        )
        return cached

    errors: list[str] = []
    with singleflight(cache_key, should_cancel) if cache_key else nullcontext():
        cached = _read_cached_expansion(cache_key, count) if cache_key else None
        if cached is not None:
            context.increment_counter("expansion_cache_hits")
            context.record_cache_hit(
                provider="gemini",
                operation="expansion",
                metadata={"cache_key": cache_key},
            )
            return cached
        try:
            kwargs = {
                "model": PRACTICE_FAST_EXPAND_MODEL,
                "level": level,
                "should_cancel": should_cancel,
            }
            if context is not None:
                kwargs["context"] = context
            raw = _practice_fast_gemini_raw(topic, count, **kwargs)
            parsed = _PracticeFastExpansion.model_validate_json(raw)
            corrected = " ".join(str(parsed.corrected or topic).split()) or topic
            queries = _normalize([corrected, *parsed.queries], count)
            if not queries:
                raise ValueError("Gemini returned no usable search queries")
            raise_if_cancelled(should_cancel)
            result = {
                "corrected": corrected,
                "queries": queries,
                "provider_used": "gemini",
            }
            if cache_key:
                _write_cached_expansion(cache_key, result)
            return result
        except CancellationError:
            raise
        except Exception as exc:
            raise_if_cancelled(should_cancel)
            errors.append(f"{PRACTICE_FAST_EXPAND_MODEL}: {exc}")
    logger.info(
        "practice-fast Gemini expansion unavailable; using literal fallback: %s",
        "; ".join(errors),
    )
    return literal_fallback(topic, count)


def expand_query(
    topic: str,
    n: int,
    level: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    raise_if_cancelled(should_cancel)
    result = deterministic_expand(topic, n, level=level)
    raise_if_cancelled(should_cancel)
    return result
