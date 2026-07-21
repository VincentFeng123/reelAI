"""Mechanical identity and validation for Gemini-grounded request facets."""
from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from typing import Any


INTENT_OBLIGATION_CONTRACT_VERSION = "intent_obligation_v1"
INTENT_OBLIGATION_KINDS = frozenset({
    "subject",
    "task",
    "relationship",
    "scope",
    "format",
    "outcome",
})
MAX_INTENT_OBLIGATIONS = 16


def _clean_text(value: object, limit: int) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())[:limit]


def _source_start(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 1_000_000 else None


def intent_obligation_key(
    source_phrase: object,
    source_start: object = 0,
) -> str:
    """Return a stable opaque key from one exact, positioned request span."""
    # Source phrases are copied from the request. Preserve case because symbols
    # and identifiers such as C/c may name different requested concepts. The
    # character offset distinguishes repeated phrases without depending on the
    # model's enum choice or requirement wording.
    phrase = _clean_text(source_phrase, 160)
    start = _source_start(source_start)
    if not phrase or start is None:
        return ""
    identity = f"{start}\0{phrase}".encode("utf-8")
    return f"io:{hashlib.sha256(identity).hexdigest()[:24]}"


def intent_obligation(
    *,
    kind: object,
    source_phrase: object,
    requirement: object,
    evidence_quote: object = "",
    source_start: object = 0,
) -> dict[str, Any] | None:
    clean_kind = _clean_text(kind, 24).casefold()
    clean_source = _clean_text(source_phrase, 160)
    clean_requirement = _clean_text(requirement, 240)
    clean_evidence = _clean_text(evidence_quote, 240)
    clean_source_start = _source_start(source_start)
    key = intent_obligation_key(clean_source, clean_source_start)
    if clean_kind not in INTENT_OBLIGATION_KINDS or not key or not clean_requirement:
        return None
    item = {
        "key": key,
        "kind": clean_kind,
        "source_phrase": clean_source,
        "source_start": clean_source_start,
        "requirement": clean_requirement,
    }
    if clean_evidence:
        item["evidence_quote"] = clean_evidence
    return item


def normalize_intent_obligations(
    value: object,
    *,
    require_evidence: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value[:MAX_INTENT_OBLIGATIONS]:
        if not isinstance(raw, dict):
            continue
        item = intent_obligation(
            kind=raw.get("kind"),
            source_phrase=raw.get("source_phrase"),
            source_start=raw.get("source_start"),
            requirement=raw.get("requirement"),
            evidence_quote=raw.get("evidence_quote"),
        )
        if item is None or (require_evidence and not item.get("evidence_quote")):
            continue
        if str(raw.get("key") or "").strip() != item["key"]:
            continue
        if item["key"] in seen:
            continue
        seen.add(item["key"])
        normalized.append(item)
    return normalized


def intent_obligation_keys(items: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("key") or "").strip()
        for item in items
        if str(item.get("key") or "").strip()
    }
