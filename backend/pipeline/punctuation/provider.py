"""Provider abstraction for punctuation inference.

The default implementation reuses ``backend/llm.py:llm_json`` — the project's provider switch
(Gemini ``response_schema`` / Groq ``json_schema`` → ``json_object`` fallback, with malformed-output
retry) — so we don't reimplement vendor dispatch. Tests inject a ``FakeProvider`` instead, keeping
the whole suite offline. Adding OpenAI / a local model later means adding a class here, not touching
the pipeline.
"""
from __future__ import annotations

from typing import Optional, Protocol

from .types import ChunkEdits


class PunctuationProvider(Protocol):
    def infer(self, system: str, user: str, *, est_tokens: int = 0) -> ChunkEdits:
        """Return the model's sparse edits for one prompt. Must raise on transport/parse failure."""
        ...


class LLMPunctuationProvider:
    """Wraps ``llm.llm_json`` with a fixed, low temperature for stable punctuation."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = 0.0) -> None:
        self.provider = provider or None
        self.model = model or None
        self.temperature = temperature

    def infer(self, system: str, user: str, *, est_tokens: int = 0) -> ChunkEdits:
        from ... import config
        from ...llm import llm_json

        # PUNCTUATION_MODEL overrides the provider default only for Gemini (config.GEMINI_MODEL);
        # Groq model choice lives in llm_json. We set it via a temporary override when provided.
        prev_model = None
        if self.model and (self.provider or config.LLM_PROVIDER).lower() == "gemini":
            prev_model, config.GEMINI_MODEL = config.GEMINI_MODEL, self.model
        try:
            return llm_json(system, user, ChunkEdits, temperature=self.temperature,
                            est_tokens=est_tokens, provider=self.provider)
        finally:
            if prev_model is not None:
                config.GEMINI_MODEL = prev_model
