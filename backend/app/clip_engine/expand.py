"""Query expansion for the existing planner and the opt-in practice-fast path.

Topic correction, aliases, and semantic expansion are owned by the persisted
TopicExpansionService. This layer performs no provider calls and spends no model
credits for the existing ``expand_query`` API; it only adds stable YouTube-oriented
query templates. ``expand_query_practice_fast`` is a separate reference path that
uses one Gemini Flash call and falls back to those deterministic templates.
"""
from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from . import config
from .cancellation import raise_if_cancelled, run_cancellable
from .errors import CancellationError

if TYPE_CHECKING:
    from .provider_runtime import GenerationContext

logger = logging.getLogger(__name__)

PRACTICE_FAST_EXPAND_MODEL = "gemini-3.5-flash"
PRACTICE_FAST_EXPAND_FALLBACK_MODEL = "gemini-3.1-pro-preview"


class _PracticeFastExpansion(BaseModel):
    corrected: str
    queries: list[str]


_PRACTICE_FAST_SYSTEM = """You expand a user's search topic into a diverse set of
YouTube search queries that maximize topical coverage.

Do this:
1. Spellcheck and correct the user's input.
2. Infer the most likely intent or sense.
3. Produce up to N distinct search queries covering the corrected topic, close synonyms,
   important sub-topics, and useful educational phrase variants such as tutorials or explainers.

Return only the requested JSON object. Put the corrected topic first in queries."""


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
) -> str:
    """Make the path's single provider call. There is intentionally no model fallback."""
    raise_if_cancelled(should_cancel)
    key = config.require_gemini_key()
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    level_text = str(level or "").strip()
    level_line = f"\nViewer level: {level_text}" if level_text else ""
    user = (
        f"User topic: {topic!r}\nN = {max(1, int(n))}{level_line}\n"
        "Return corrected and queries as JSON."
    )
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=_PRACTICE_FAST_SYSTEM,
                response_mime_type="application/json",
                response_json_schema=_PracticeFastExpansion.model_json_schema(),
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        raise_if_cancelled(should_cancel)
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
) -> str:
    return run_cancellable(
        lambda: _practice_fast_gemini_raw_async(
            topic,
            n,
            model=model,
            level=level,
            should_cancel=should_cancel,
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
    """Practice-compatible Gemini expansion with deterministic fail-soft behavior.

    ``context`` is accepted for direct compatibility with the live discovery call.
    Expansion deliberately does not reserve the Supadata search budget tracked by it.
    """
    del context
    topic = " ".join(str(topic or "").split())
    count = max(0, int(n))
    if not topic or count == 0:
        return deterministic_expand(topic, count, level=level)
    raise_if_cancelled(should_cancel)
    errors: list[str] = []
    for model in (PRACTICE_FAST_EXPAND_MODEL, PRACTICE_FAST_EXPAND_FALLBACK_MODEL):
        try:
            raw = _practice_fast_gemini_raw(
                topic,
                count,
                model=model,
                level=level,
                should_cancel=should_cancel,
            )
            parsed = _PracticeFastExpansion.model_validate_json(raw)
            corrected = " ".join(str(parsed.corrected or topic).split()) or topic
            queries = _normalize([corrected, *parsed.queries], count)
            if not queries:
                raise ValueError("Gemini returned no usable search queries")
            raise_if_cancelled(should_cancel)
            return {
                "corrected": corrected,
                "queries": queries,
                "provider_used": "gemini",
            }
        except CancellationError:
            raise
        except Exception as exc:
            raise_if_cancelled(should_cancel)
            errors.append(f"{model}: {exc}")
    logger.info(
        "practice-fast Gemini expansion unavailable; using deterministic fallback: %s",
        "; ".join(errors),
    )
    return deterministic_expand(topic, count, level=level)


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
