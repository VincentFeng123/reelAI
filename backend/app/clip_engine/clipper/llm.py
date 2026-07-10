"""One provider-abstracted structured-JSON LLM call.

Every structure-first pass (content-type detection, unit extraction, role labeling,
concept extraction, dependency refinement, the clip-only judge, context cards) calls
Gemini via ``generate_json``, validates the Pydantic ``response_schema`` response, and
retries once with a stricter JSON instruction on malformed output. Returns a *validated*
Pydantic instance.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from . import config
from ..provider_runtime import GenerationContext
from ..errors import ProviderRequestError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StructuredResult:
    value: T
    model_used: str
    quality_degraded: bool


def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _estimate_tokens(text: str) -> int:
    # Reuse select.estimate_tokens (tiktoken with a char fallback) without a hard dep.
    try:
        from .pipeline.select import estimate_tokens
        return estimate_tokens(text)
    except Exception:
        import math
        return math.ceil(len(text) / config.CHARS_PER_TOKEN)


def llm_json_result(
    system: str,
    user: str,
    schema: type[T],
    *,
    temperature: float = 0.2,
    est_tokens: int = 0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: int = 8192,
    should_cancel: Callable[[], bool] | None = None,
    fallback_model: Optional[str] = None,
    context: GenerationContext | None = None,
) -> StructuredResult:
    """Return a validated ``schema`` instance from Gemini.

    One structured call (Pydantic ``response_schema``) followed by local
    validation. Malformed output is not retried because only transient provider
    failures are eligible for provider retries.
    """
    # provider param kept for API compatibility; only "gemini" is supported.
    est = est_tokens or (_estimate_tokens(system + user) + config.EXPECTED_OUTPUT_TOKENS)  # noqa: F841
    from . import gemini_client
    try:
        response = gemini_client.generate_json_result(
            system, user, schema, temperature=temperature,
            model=model, max_output_tokens=max_output_tokens,
            should_cancel=should_cancel, context=context,
        )
    except Exception as e:
        fallback = str(fallback_model or "").strip()
        if not (fallback and fallback != model and gemini_client.is_model_unavailable(e)):
            raise
        logger.warning("model %s unavailable (%s); falling back to %s",
                       model, e, fallback)
        model = fallback
        response = gemini_client.generate_json_result(
            system, user, schema, temperature=temperature,
            model=model, max_output_tokens=max_output_tokens,
            should_cancel=should_cancel, context=context, quality_degraded=True,
        )
    try:
        value = schema.model_validate_json(_strip_fences(response.text))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderRequestError(
            "Gemini returned malformed structured segmentation output.",
            provider="gemini",
            operation="segmentation",
            detail=str(exc)[:500],
        ) from exc
    return StructuredResult(value, response.model_used, response.quality_degraded)


def llm_json(
    system: str,
    user: str,
    schema: type[T],
    *,
    temperature: float = 0.2,
    est_tokens: int = 0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: int = 8192,
    should_cancel: Callable[[], bool] | None = None,
    fallback_model: Optional[str] = None,
    context: GenerationContext | None = None,
) -> T:
    return llm_json_result(
        system,
        user,
        schema,
        temperature=temperature,
        est_tokens=est_tokens,
        provider=provider,
        model=model,
        max_output_tokens=max_output_tokens,
        should_cancel=should_cancel,
        fallback_model=fallback_model,
        context=context,
    ).value
