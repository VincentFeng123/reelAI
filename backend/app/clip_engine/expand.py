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
import re
import unicodedata
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import config
from .cancellation import raise_if_cancelled, run_cancellable
from .errors import CancellationError
from .segment_cache import SEGMENT_CACHE_TTL_SEC
from .singleflight import singleflight
from ..db import dumps_json, fetch_one, get_conn, now_iso, upsert

if TYPE_CHECKING:
    from .provider_runtime import GenerationContext

logger = logging.getLogger(__name__)

_INTENT_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_INTENT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "including", "please", "the", "through", "to", "what", "when", "where",
    "which", "who", "why", "with", "would", "you",
})

PRACTICE_FAST_EXPAND_MODEL = "gemini-3.1-flash-lite"
# Gemini rejects manually configured deadlines below ten seconds.
PRACTICE_FAST_EXPAND_TIMEOUT_MS = 10_000
PRACTICE_FAST_EXPAND_OUTPUT_TOKENS = 2_048
PRACTICE_FAST_EXPAND_CACHE_VERSION = 7
# An expansion can be nearly one segment-cache lifetime old when it discovers a
# newly analyzed source. Keeping it for two lifetimes guarantees that source's
# subsequent valid segment-cache lifetime never triggers another expansion call.
PRACTICE_FAST_EXPAND_CACHE_TTL_SEC = 2 * SEGMENT_CACHE_TTL_SEC


class _PracticeFastIntentConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(min_length=1, max_length=32)
    kind: Literal[
        "subject",
        "task",
        "relationship",
        "scope",
        "format",
        "outcome",
    ]
    source_phrase: str = Field(min_length=1, max_length=160)
    requirement: str = Field(min_length=1, max_length=240)


class _PracticeFastQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=240)
    preserved_constraint_ids: list[str] = Field(min_length=1, max_length=16)


class _PracticeFastExpansion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corrected: str
    intent_constraints: list[_PracticeFastIntentConstraint] = Field(
        min_length=1,
        max_length=16,
    )
    queries: list[_PracticeFastQuery]

    @model_validator(mode="after")
    def _unique_intent_ids(self):
        ids = [constraint.constraint_id for constraint in self.intent_constraints]
        if len(set(ids)) != len(ids):
            raise ValueError("intent constraint ids must be unique")
        return self


_PRACTICE_FAST_SYSTEM = """You expand a user's search topic into a diverse set of
YouTube search queries that maximize topical coverage.

Do this:
1. Spellcheck and correct the user's input.
2. Infer the most likely intent or sense.
3. Produce up to N concise queries that a person would actually search on YouTube.
4. Preserve the user's named subject in every query. Cover close synonyms, genuinely related
   informational facets, useful prerequisites, and educational sources, but never redirect the
   search into a merely adjacent field.
5. In the optimized query list, prioritize substantive spoken lessons, lectures, or worked
   explanations with natural pauses and little or no background music. Avoid Shorts, reaction
   videos, compilations, montages, and explanations that only work when unseen visuals are shown.
   Prefer focused teaching videos. Never query for a full course, complete course, marathon,
   playlist, or lecture series unless the user explicitly requests that format.
6. Before writing queries, decompose the exact request into atomic mandatory intent constraints.
   Give every named subject, requested operation or task, requested relationship, scope qualifier,
   requested format, and requested outcome its own constraint and label its kind. Give every named
   member of a list its own separate constraint; the schema supports sixteen. Copy the exact words
   that introduced each constraint into source_phrase. Do not collapse a requested task or facet
   into the subject. SUBJECT means the governing named topic, law, concept, or object; components
   and named list members under that governing topic are SCOPE constraints. For
   "Explain Newton's second law F=ma with net force, mass, acceleration, units, and solving for
   each variable," Newton's second law F=ma is SUBJECT,
   Explain is TASK, net force, mass, acceleration, and units are four separate SCOPE constraints,
   and solving for each variable is OUTCOME.
7. The first broad query must preserve every constraint. Every later focused query must preserve
   every subject constraint plus one or more distinct task, relationship, scope, format, or outcome
   constraints. Collectively target every named facet or list member; when the request says each,
   every, or all, give distinct members focused coverage where N permits.
8. For every query, return exactly the IDs of the constraints it preserves. Synonyms and natural
   YouTube wording are welcome, but focused queries must not lose the governing subject or drift
   into an adjacent field.

Return only the requested JSON object with corrected, intent_constraints, and queries. Keep
corrected separate from queries. Return N optimized search queries; do not automatically spend
one query on the raw literal wording."""


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
    if not corrected or not queries:
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


def _intent_tokens(value: object) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return [match.group(0) for match in _INTENT_TOKEN_RE.finditer(normalized)]


def _contains_token_phrase(text: object, phrase: object) -> bool:
    source = _intent_tokens(text)
    target = _intent_tokens(phrase)
    if not source or not target or len(target) > len(source):
        return False
    width = len(target)
    return any(
        source[index : index + width] == target
        for index in range(len(source) - width + 1)
    )


def _contains_ordered_topic_terms(text: object, phrase: object) -> bool:
    """Accept exact spans or grounded in-order terms without stitched invention."""

    if _contains_token_phrase(text, phrase):
        return True
    source = [
        token for token in _intent_tokens(text) if token not in _INTENT_STOPWORDS
    ]
    target = [
        token for token in _intent_tokens(phrase) if token not in _INTENT_STOPWORDS
    ]
    if not source or not target:
        return False
    cursor = 0
    for token in source:
        if token == target[cursor]:
            cursor += 1
            if cursor == len(target):
                return True
    return False


def _covering_focus_indices(
    focused: list[tuple[str, frozenset[str]]],
    required_ids: set[str],
    limit: int,
) -> tuple[int, ...] | None:
    """Find the earliest smallest bounded query set with complete facet coverage."""

    target = frozenset(required_ids)
    if not target:
        return ()
    states: dict[frozenset[str], tuple[int, ...]] = {frozenset(): ()}
    for index, (_query, signature) in enumerate(focused):
        updates: dict[frozenset[str], tuple[int, ...]] = {}
        for covered, selected in tuple(states.items()):
            if len(selected) >= limit:
                continue
            combined = covered | signature
            candidate = (*selected, index)
            previous = updates.get(combined, states.get(combined))
            if previous is None or (len(candidate), candidate) < (
                len(previous),
                previous,
            ):
                updates[combined] = candidate
        states.update(updates)
    return states.get(target)


def _validated_ai_queries(
    topic: str,
    parsed: _PracticeFastExpansion,
    count: int,
) -> list[str]:
    """Keep one exact-request query plus subject-grounded facet queries."""
    constraints = list(parsed.intent_constraints)
    constraint_ids = {constraint.constraint_id for constraint in constraints}
    if not constraint_ids:
        return []
    subject_ids = {
        constraint.constraint_id
        for constraint in constraints
        if constraint.kind == "subject"
    }
    if not subject_ids:
        return []
    if any(
        not _contains_ordered_topic_terms(topic, constraint.source_phrase)
        for constraint in constraints
    ):
        return []
    required_topic_tokens = {
        token for token in _intent_tokens(topic) if token not in _INTENT_STOPWORDS
    }
    covered_topic_tokens = {
        token
        for constraint in constraints
        for token in _intent_tokens(constraint.source_phrase)
        if token not in _INTENT_STOPWORDS
    }
    if required_topic_tokens and not required_topic_tokens.issubset(covered_topic_tokens):
        return []

    broad: list[str] = []
    focused: list[tuple[str, frozenset[str]]] = []
    focused_signatures: set[frozenset[str]] = set()
    non_subject_ids = constraint_ids - subject_ids
    for query in parsed.queries:
        preserved = {
            " ".join(str(value or "").split())
            for value in query.preserved_constraint_ids
            if " ".join(str(value or "").split())
        }
        if (
            not preserved
            or not preserved.issubset(constraint_ids)
            or not subject_ids.issubset(preserved)
        ):
            continue
        if preserved == constraint_ids:
            broad.append(query.text)
            continue
        signature = frozenset(preserved & non_subject_ids)
        if not signature or signature in focused_signatures:
            continue
        focused_signatures.add(signature)
        focused.append((query.text, signature))

    normalized_model_broad = _normalize(broad, len(broad))
    primary_broad = normalized_model_broad[0] if normalized_model_broad else topic
    broad_keys = {
        _key(query) for query in [topic, *normalized_model_broad]
    }
    normalized_focused: list[tuple[str, frozenset[str]]] = []
    focused_keys: set[str] = set()
    for query, signature in focused:
        text = " ".join(str(query or "").split())
        key = _key(text)
        if not text or not key or key in broad_keys or key in focused_keys:
            continue
        focused_keys.add(key)
        normalized_focused.append((text, signature))
    if not non_subject_ids:
        return _normalize(
            [primary_broad, *normalized_model_broad[1:]],
            count,
        )
    if len(non_subject_ids) <= 2:
        normalized_focused.extend(
            (query, frozenset(non_subject_ids))
            for query in normalized_model_broad[1:]
        )
    if count <= 1:
        return _normalize([primary_broad], count)
    if count > 1 and non_subject_ids and not normalized_focused:
        return []

    focused_limit = max(0, count - 1)
    selected_indices = _covering_focus_indices(
        normalized_focused,
        non_subject_ids,
        focused_limit,
    )
    if selected_indices is None:
        return []
    selected_index_set = set(selected_indices)
    for index in range(len(normalized_focused)):
        if len(selected_index_set) >= focused_limit:
            break
        selected_index_set.add(index)
    selected_focused = [
        normalized_focused[index]
        for index in sorted(selected_index_set)
    ]
    return _normalize(
        [
            primary_broad,
            *(query for query, _signature in selected_focused),
        ],
        count,
    )


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
        "Return corrected, intent_constraints, and queries as JSON."
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
                    "cached_content_token_count": usage_field(
                        "cached_content_token_count"
                    ),
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
            queries = _validated_ai_queries(topic, parsed, count)
            if not queries:
                raise ValueError(
                    "Gemini returned no search query preserving the exact intent contract"
                )
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
