"""Knowledge-level semantics: level names, difficulty-scale mapping, and the
effective serving target. Spec: docs/superpowers/specs/2026-07-08-knowledge-level-design.md.

The level lives on the materials row (`knowledge_level`, `level_adjustment`);
everything here is pure — no DB, no LLM.
"""
from __future__ import annotations

KNOWLEDGE_LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced")

# Positions on the same 0..1 scale the engine's per-clip `difficulty` uses.
LEVEL_VALUES: dict[str, float] = {
    "beginner": 0.15,
    "intermediate": 0.50,
    "advanced": 0.85,
}

# Auto-adjust drift can never exceed one level step; the user's explicit
# choice stays authoritative.
ADJUSTMENT_BOUND: float = 0.35


def normalize_knowledge_level(value: str | None) -> str:
    """Lowercased/stripped level name; absent -> 'beginner'; unknown -> ValueError."""
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return "beginner"
    if cleaned not in KNOWLEDGE_LEVELS:
        raise ValueError(f"unknown knowledge_level: {value!r}")
    return cleaned


def effective_level_target(level: str | None, adjustment: float | None) -> float:
    """The difficulty the feed should aim at RIGHT NOW for this material."""
    base = LEVEL_VALUES[normalize_knowledge_level(level)]
    adj = max(-ADJUSTMENT_BOUND, min(ADJUSTMENT_BOUND, float(adjustment or 0.0)))
    return max(0.0, min(1.0, base + adj))
