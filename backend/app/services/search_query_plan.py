"""Validated, cached AI search plans for topic-driven video retrieval.

The learner's literal topic remains the identity and highest-trust query. One
structured LLM expansion may add search coverage; no hand-authored curriculum,
public-knowledge vocabulary, or deterministic query templates are mixed in.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..clip_engine.cancellation import raise_if_cancelled
from ..db import dumps_json, fetch_one, now_iso, upsert
from . import llm_router
from .embeddings import EmbeddingService


PLAN_VERSION = 2
PLAN_TTL_SEC = 24 * 60 * 60
FALLBACK_PLAN_TTL_SEC = 15 * 60

_GENERIC_QUERY_WORDS = {
    "a", "an", "and", "basics", "beginner", "beginners", "course", "deep",
    "dive", "explained", "for", "fundamentals", "how", "introduction", "lecture",
    "of", "overview", "the", "to", "tutorial", "works",
}
_GENERIC_TERMS = {
    "basics", "foundations", "introduction", "overview", "problem solving",
    "worked examples", "tutorial", "lecture", "course",
}
_LOW_VALUE_INTENT_PATTERNS = (
    re.compile(r"\bcollege admissions?\b", re.I),
    re.compile(r"\bcollege applications?\b", re.I),
    re.compile(r"\b(?:ap|college) class(?:es)? rank(?:ed|ing)?\b", re.I),
    re.compile(r"\bbest (?:ap|college) class(?:es)?\b", re.I),
    re.compile(r"\bworst (?:ap|college) class(?:es)?\b", re.I),
)


def normalize_query(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("_", " ")
    text = re.sub(r"[^\w\+# ]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _clean(value: object, *, max_length: int = 160) -> str:
    return " ".join(str(value or "").split()).strip()[:max_length]


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(token for token in normalize_query(value).split() if token not in _GENERIC_QUERY_WORDS)


def _stem(token: str) -> str:
    for suffix in ("ization", "ational", "ations", "ation", "ments", "ment", "ives", "ive", "ing", "ies", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def semantic_query_family(value: object) -> str:
    """Collapse pedagogy templates without collapsing real subtopics."""
    tokens = [_stem(token) for token in _tokens(value)]
    return " ".join(tokens) or normalize_query(value)


class AIQueryExpansion(BaseModel):
    """Gemini response schema; application validation remains mandatory."""

    model_config = ConfigDict(extra="forbid")

    canonical_query: str = Field(max_length=160)
    literal_is_ambiguous: bool
    aliases: list[str] = Field(max_length=8)
    subtopics: list[str] = Field(max_length=16)
    related_terms: list[str] = Field(max_length=8)


class PlannedSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    family: str
    provenance: str
    trust: Literal["literal", "canonical", "ai"]


class SearchQueryPlan(BaseModel):
    """Internal retrieval contract persisted as JSON in ``llm_cache``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = PLAN_VERSION
    literal_query: str
    canonical_query: str
    accepted_aliases: list[str] = Field(default_factory=list)
    accepted_subtopics: list[str] = Field(default_factory=list)
    accepted_related_terms: list[str] = Field(default_factory=list)
    trusted_signature: list[str] = Field(default_factory=list)
    literal_is_ambiguous: bool = False
    provenance: dict[str, list[str]] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)
    queries: list[PlannedSearchQuery] = Field(default_factory=list)
    ai_status: Literal["validated", "unavailable", "invalid"] = "unavailable"

    def query_window(self, *, offset: int, limit: int) -> list[PlannedSearchQuery]:
        start = max(0, int(offset))
        count = max(0, int(limit))
        return list(self.queries[start : start + count])

    def as_topic_expansion(self) -> dict[str, Any]:
        return {
            "canonical_topic": self.canonical_query,
            "aliases": list(self.accepted_aliases),
            "subtopics": list(self.accepted_subtopics),
            "related_terms": list(self.accepted_related_terms),
        }


def _cache_key(literal_query: str) -> str:
    digest = hashlib.sha256(normalize_query(literal_query).encode("utf-8")).hexdigest()
    return f"search_query_plan:v{PLAN_VERSION}:{digest}"


def _age_seconds(created_at: object) -> float:
    try:
        parsed = datetime.fromisoformat(str(created_at or "").replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())


def _read_cached_plan(conn: Any, literal_query: str) -> tuple[SearchQueryPlan | None, float]:
    if conn is None:
        return None, float("inf")
    try:
        row = fetch_one(
            conn,
            "SELECT response_json, created_at FROM llm_cache WHERE cache_key = ?",
            (_cache_key(literal_query),),
        )
    except Exception:
        return None, float("inf")
    if not row:
        return None, float("inf")
    try:
        plan = SearchQueryPlan.model_validate_json(str(row.get("response_json") or "{}"))
    except (ValidationError, ValueError):
        return None, float("inf")
    if plan.version != PLAN_VERSION or normalize_query(plan.literal_query) != normalize_query(literal_query):
        return None, float("inf")
    return plan, _age_seconds(row.get("created_at"))


def _write_cached_plan(conn: Any, plan: SearchQueryPlan) -> None:
    if conn is None:
        return
    try:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": _cache_key(plan.literal_query),
                "response_json": dumps_json(plan.model_dump(mode="json")),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
    except Exception:
        # Retrieval may continue with the validated in-memory plan; the next
        # request will retry persistence rather than hiding usable results.
        return


def _append_unique(values: list[str], value: object, seen: set[str]) -> str | None:
    cleaned = _clean(value)
    key = normalize_query(cleaned)
    if not cleaned or not key or key in seen:
        return None
    seen.add(key)
    values.append(cleaned)
    return cleaned


def _ai_term_rejection(term: object) -> str | None:
    cleaned = _clean(term)
    normalized = normalize_query(cleaned)
    if not cleaned or not normalized or normalized in _GENERIC_TERMS or not _tokens(cleaned):
        return "generic or blank"
    if any(pattern.search(cleaned) for pattern in _LOW_VALUE_INTENT_PATTERNS):
        return "disallowed low-value intent"
    return None


def _lexically_coherent(anchor: object, candidate: object) -> bool:
    anchor_key = normalize_query(anchor)
    candidate_key = normalize_query(candidate)
    if not anchor_key or not candidate_key:
        return False
    if anchor_key == candidate_key or difflib.SequenceMatcher(None, anchor_key, candidate_key).ratio() >= 0.82:
        return True
    anchor_tokens = {_stem(token) for token in _tokens(anchor)}
    candidate_tokens = {_stem(token) for token in _tokens(candidate)}
    if not anchor_tokens or not candidate_tokens:
        return False
    overlap = len(anchor_tokens.intersection(candidate_tokens))
    return overlap / max(len(anchor_tokens), len(candidate_tokens)) >= 0.67


def _semantic_relevance_scores(
    anchors: list[str],
    candidates: list[str],
) -> dict[str, float] | None:
    """Score candidates with real sentence embeddings; never use hash fallback."""
    clean_anchors = [_clean(value) for value in anchors if _clean(value)]
    clean_candidates = [_clean(value) for value in candidates if _clean(value)]
    if not clean_anchors or not clean_candidates:
        return {}
    service = EmbeddingService()
    vectors = service.embed_semantic([*clean_anchors, *clean_candidates])
    if vectors is None:
        return None
    anchor_vectors = vectors[: len(clean_anchors)]
    scores: dict[str, float] = {}
    for index, candidate in enumerate(clean_candidates):
        candidate_vector = vectors[len(clean_anchors) + index]
        scores[normalize_query(candidate)] = max(
            float(anchor_vector @ candidate_vector)
            for anchor_vector in anchor_vectors
        )
    return scores


def _planned_queries(
    *,
    literal: str,
    canonical: str,
    ai_aliases: list[str],
    ai_subtopics: list[str],
    ai_related: list[str],
    provenance: dict[str, list[str]],
) -> list[PlannedSearchQuery]:
    planned: list[PlannedSearchQuery] = []
    seen: set[str] = set()
    root_family = semantic_query_family(canonical or literal)

    def add(value: str, trust: str, source: str, *, root: bool = False) -> None:
        text = _clean(value)
        key = normalize_query(text)
        if not text or not key or key in seen or len(planned) >= 12:
            return
        seen.add(key)
        family = root_family if root else semantic_query_family(text)
        planned.append(
            PlannedSearchQuery(text=text, family=family, provenance=source, trust=trust)
        )
        provenance.setdefault(key, [])
        if source not in provenance[key]:
            provenance[key].append(source)

    add(literal, "literal", "literal", root=True)
    if normalize_query(canonical) != normalize_query(literal):
        add(canonical, "canonical", "ai", root=True)

    # Gemini owns every non-literal query. Interleave its categories so fast
    # mode does not spend both expansion slots on near-identical aliases.
    for index in range(max(len(ai_aliases), len(ai_subtopics), len(ai_related))):
        if index < len(ai_aliases):
            add(ai_aliases[index], "ai", "ai", root=True)
        if index < len(ai_subtopics):
            add(ai_subtopics[index], "ai", "ai")
        if index < len(ai_related):
            add(ai_related[index], "ai", "ai")
    return planned[:12]


def build_search_query_plan(
    conn: Any,
    *,
    literal_query: str,
    should_cancel: Callable[[], bool] | None = None,
) -> SearchQueryPlan:
    """Build or load exactly one validated plan per normalized literal topic."""
    raise_if_cancelled(should_cancel)
    literal = _clean(literal_query)
    if not normalize_query(literal):
        raise ValueError("literal_query must contain non-whitespace text")

    cached, age_sec = _read_cached_plan(conn, literal)
    if cached is not None:
        ttl = PLAN_TTL_SEC if cached.ai_status == "validated" else FALLBACK_PLAN_TTL_SEC
        if age_sec < ttl:
            return cached

    provenance: dict[str, list[str]] = {normalize_query(literal): ["literal"]}
    canonical = literal
    aliases: list[str] = []
    subtopics: list[str] = []
    related: list[str] = []
    seen_aliases: set[str] = {normalize_query(literal)}
    seen_subtopics: set[str] = set()
    seen_related: set[str] = set()
    seen_ai_terms: set[str] = {normalize_query(literal)}

    system = (
        "Create a strict educational search expansion for one literal topic. "
        "Return only the requested JSON schema. Do not change domains, infer learner intent, "
        "or add rankings, admissions, entertainment, or generic pedagogy terms."
    )
    user = (
        f"Literal topic: {literal}\n"
        "Provide its canonical search query, concise aliases, concrete subtopics, "
        "and tightly related search terms. Set literal_is_ambiguous when the literal "
        "has common meanings in different subject domains. Every non-literal query "
        "used by the application will come only from this response."
    )
    ai_status: Literal["validated", "unavailable", "invalid"] = "unavailable"
    rejection_reasons: list[str] = []
    ai_aliases: list[str] = []
    ai_subtopics: list[str] = []
    ai_related: list[str] = []
    literal_is_ambiguous = False
    try:
        raw = llm_router.chat_completion(
            system=system,
            user=user,
            temperature=0.1,
            json_mode=True,
            max_tokens=900,
            response_schema=AIQueryExpansion,
            should_cancel=should_cancel,
        )
        raise_if_cancelled(should_cancel)
        if raw:
            payload = AIQueryExpansion.model_validate_json(raw)
            ai_status = "validated"
            literal_is_ambiguous = bool(payload.literal_is_ambiguous)
            ai_canonical = _clean(payload.canonical_query)
            canonical_rejection = _ai_term_rejection(ai_canonical) if ai_canonical else None
            canonical_scores = (
                {}
                if not ai_canonical or _lexically_coherent(literal, ai_canonical)
                else _semantic_relevance_scores([literal], [ai_canonical])
            )
            canonical_score = float((canonical_scores or {}).get(normalize_query(ai_canonical), 0.0))
            canonical_coherent = (
                not ai_canonical
                or _lexically_coherent(literal, ai_canonical)
                or canonical_score >= 0.34
            )
            if ai_canonical and canonical_rejection is None and canonical_coherent:
                canonical = ai_canonical
                provenance.setdefault(normalize_query(canonical), []).append("ai")
                seen_aliases.add(normalize_query(canonical))
                seen_ai_terms.add(normalize_query(canonical))
            elif ai_canonical:
                reason = canonical_rejection or "not semantically anchored to literal topic"
                rejection_reasons.append(f"canonical_query: {reason}")

            raw_term_groups = (
                ("aliases", payload.aliases, aliases, ai_aliases, seen_aliases, 0.34),
                ("subtopics", payload.subtopics, subtopics, ai_subtopics, seen_subtopics, 0.28),
                ("related_terms", payload.related_terms, related, ai_related, seen_related, 0.30),
            )
            semantic_candidates = [
                _clean(term)
                for _field, values, _destination, _ai_destination, _seen, _threshold in raw_term_groups
                for term in values
                if _ai_term_rejection(term) is None
                and not any(_lexically_coherent(anchor, term) for anchor in (literal, canonical))
            ]
            semantic_scores = _semantic_relevance_scores(
                [literal, canonical],
                semantic_candidates,
            ) if semantic_candidates else {}

            for field_name, values, destination, ai_destination, seen, threshold in raw_term_groups:
                for index, raw_term in enumerate(values):
                    term = _clean(raw_term)
                    rejection = _ai_term_rejection(term)
                    if rejection is not None:
                        rejection_reasons.append(f"{field_name}[{index}]: {rejection}")
                        continue
                    if normalize_query(term) in seen_ai_terms:
                        continue
                    lexical_match = any(
                        _lexically_coherent(anchor, term)
                        for anchor in (literal, canonical)
                    )
                    semantic_score = float((semantic_scores or {}).get(normalize_query(term), 0.0))
                    if not lexical_match and semantic_score < threshold:
                        rejection_reasons.append(
                            f"{field_name}[{index}]: not semantically anchored to literal topic"
                        )
                        continue
                    accepted = _append_unique(destination, term, seen)
                    if not accepted:
                        continue
                    seen_ai_terms.add(normalize_query(accepted))
                    ai_destination.append(accepted)
                    provenance.setdefault(normalize_query(accepted), []).append("ai")
            if (
                normalize_query(canonical) == normalize_query(literal)
                and not aliases
                and not subtopics
                and not related
            ):
                ai_status = "invalid"
                rejection_reasons.append("ai_expansion: response contained no usable expansion")
        else:
            rejection_reasons.append("ai_expansion: no callable model returned a plan")
    except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
        ai_status = "invalid"
        rejection_reasons.append("ai_expansion: response failed schema or semantic validation")

    # A stale validated plan is safer than replacing it after a transient model
    # outage. It has already passed both schema and application validation.
    if ai_status != "validated" and cached is not None and cached.ai_status == "validated":
        return cached

    signature: list[str] = []
    signature_seen: set[str] = set()
    for raw in [literal, canonical, *aliases, *subtopics, *related]:
        _append_unique(signature, raw, signature_seen)

    planned = _planned_queries(
        literal=literal,
        canonical=canonical,
        ai_aliases=ai_aliases,
        ai_subtopics=ai_subtopics,
        ai_related=ai_related,
        provenance=provenance,
    )
    plan = SearchQueryPlan(
        literal_query=literal,
        canonical_query=canonical,
        accepted_aliases=aliases[:8],
        accepted_subtopics=subtopics[:16],
        accepted_related_terms=related[:8],
        trusted_signature=signature[:32],
        literal_is_ambiguous=literal_is_ambiguous,
        provenance=provenance,
        rejection_reasons=rejection_reasons[:32],
        queries=planned,
        ai_status=ai_status,
    )
    raise_if_cancelled(should_cancel)
    _write_cached_plan(conn, plan)
    return plan


def topic_signature_evidence(text: str, plan: SearchQueryPlan) -> list[str]:
    """Return accepted plan terms found in an exact validated transcript window.

    Hash embeddings are never consulted. When Gemini marks the literal as
    cross-domain ambiguous, matching the root alone is insufficient: the window
    must also contain one of its more specific expansion terms.
    """
    cleaned = _clean(text, max_length=20_000)
    if not cleaned or any(pattern.search(cleaned) for pattern in _LOW_VALUE_INTENT_PATTERNS):
        return []
    text_tokens = {_stem(token) for token in normalize_query(cleaned).split()}
    if not text_tokens:
        return []

    matches: list[str] = []
    for term in plan.trusted_signature:
        term_tokens = {_stem(token) for token in _tokens(term)}
        if not term_tokens:
            continue
        overlap = term_tokens.intersection(text_tokens)
        required = 1 if len(term_tokens) == 1 else max(2, (2 * len(term_tokens) + 2) // 3)
        if len(overlap) >= required:
            matches.append(term)

    if plan.literal_is_ambiguous:
        root_families = {
            semantic_query_family(plan.literal_query),
            semantic_query_family(plan.canonical_query),
        }
        specific_ai_match = any(
            "ai" in plan.provenance.get(normalize_query(term), [])
            and semantic_query_family(term) not in root_families
            for term in matches
        )
        if not specific_ai_match:
            return []
    return matches


def transcript_window_matches_topic(text: str, plan: SearchQueryPlan) -> bool:
    return bool(topic_signature_evidence(text, plan))


__all__ = [
    "AIQueryExpansion",
    "PlannedSearchQuery",
    "SearchQueryPlan",
    "build_search_query_plan",
    "normalize_query",
    "semantic_query_family",
    "topic_signature_evidence",
    "transcript_window_matches_topic",
]
