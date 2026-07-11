"""One provider-abstracted structured-JSON LLM call.

Every structure-first pass (content-type detection, unit extraction, role labeling,
concept extraction, dependency refinement, the clip-only judge, context cards) needs
the same dance the single-pass selector already does in ``select.run_llm_selection``:
ask Gemini (Pydantic ``response_schema``) or Groq (OpenAI-compatible ``json_schema`` →
``json_object`` fallback), validate, and retry once with a stricter instruction on
malformed output. This lifts that logic into a single entry point so no pass has to
re-implement it, and returns a *validated* Pydantic instance.
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
    """A strict 'return only JSON of this shape' instruction for the Groq json_object path."""
    return (
        "Respond with ONLY a single JSON object (no prose, no markdown fences) that "
        "conforms to this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )


def _groq_json_schema(schema: type[BaseModel]) -> dict:
    """Best-effort OpenAI-compatible json_schema wrapper around a Pydantic model."""
    js = schema.model_json_schema()
    js.setdefault("additionalProperties", False)
    return {"type": "json_schema",
            "json_schema": {"name": schema.__name__.lower(), "strict": False, "schema": js}}


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
    """Return a validated ``schema`` instance from the configured LLM provider.

    Gemini: one structured call (Pydantic ``response_schema``) → validate → on failure,
    retry once with an explicit JSON instruction. Groq: try ``json_schema`` → fall back
    to ``json_object`` with ``FALLBACK_RETRIES`` correction rounds. Raises on final failure.
    """
    prov = (provider or config.LLM_PROVIDER or "gemini").lower()
    est = est_tokens or (_estimate_tokens(system + user) + config.EXPECTED_OUTPUT_TOKENS)

    if prov == "gemini":
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

    # ── Groq (OpenAI-compatible) ────────────────────────────────────────────
    from .groq_client import chat
    try:
        raw = chat(config.LLM_PRIMARY, system, user,
                   response_format=_groq_json_schema(schema),
                   temperature=temperature, est_tokens=est)
        return schema.model_validate_json(_strip_fences(raw))
    except Exception:
        pass

    fb = user + "\n\n" + _json_instruction(schema)
    last: Optional[Exception] = None
    for _ in range(config.FALLBACK_RETRIES + 1):
        raw = chat(config.LLM_FALLBACK, system, fb,
                   response_format={"type": "json_object"},
                   temperature=temperature, est_tokens=est)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            last = e
            fb = (user + "\n\n" + _json_instruction(schema)
                  + f"\n\nPrevious reply was invalid ({str(e)[:120]}). Return corrected JSON only.")
    raise RuntimeError(f"llm_json failed for {schema.__name__}: {last}")
