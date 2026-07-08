"""One provider-abstracted structured-JSON LLM call.

Every structure-first pass (content-type detection, unit extraction, role labeling,
concept extraction, dependency refinement, the clip-only judge, context cards) calls
Gemini via ``generate_json``, validates the Pydantic ``response_schema`` response, and
retries once with a stricter JSON instruction on malformed output. Returns a *validated*
Pydantic instance.
"""
from __future__ import annotations

import json
import re
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)


def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _json_instruction(schema: type[BaseModel]) -> str:
    """A strict 'return only JSON of this shape' instruction appended on retry."""
    return (
        "Respond with ONLY a single JSON object (no prose, no markdown fences) that "
        "conforms to this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )


def _estimate_tokens(text: str) -> int:
    # Reuse select.estimate_tokens (tiktoken with a char fallback) without a hard dep.
    try:
        from .pipeline.select import estimate_tokens
        return estimate_tokens(text)
    except Exception:
        import math
        return math.ceil(len(text) / config.CHARS_PER_TOKEN)


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
) -> T:
    """Return a validated ``schema`` instance from Gemini.

    One structured call (Pydantic ``response_schema``) → validate → on failure,
    retry once with an explicit JSON instruction. Raises on final failure.
    """
    # provider param kept for API compatibility; only "gemini" is supported.
    est = est_tokens or (_estimate_tokens(system + user) + config.EXPECTED_OUTPUT_TOKENS)  # noqa: F841
    from . import gemini_client
    raw = gemini_client.generate_json(system, user, schema, temperature=temperature,
                                      model=model, max_output_tokens=max_output_tokens)
    try:
        return schema.model_validate_json(_strip_fences(raw))
    except (ValidationError, json.JSONDecodeError, ValueError):
        raw = gemini_client.generate_json(
            system, user + "\n\n" + _json_instruction(schema), schema, temperature=temperature,
            model=model, max_output_tokens=max_output_tokens,
        )
        return schema.model_validate_json(_strip_fences(raw))
