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
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...concept_tokens import semantic_key, semantic_tokens
from . import config
from .cancellation import raise_if_cancelled, run_cancellable
from .errors import CancellationError, ProviderConfigurationError
from .segment_cache import SEGMENT_CACHE_TTL_SEC
from .singleflight import singleflight
from ..db import dumps_json, fetch_one, get_conn, now_iso, upsert

if TYPE_CHECKING:
    from .provider_runtime import GenerationContext

logger = logging.getLogger(__name__)

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
PRACTICE_FAST_EXPAND_ATTEMPTS = 2
PRACTICE_FAST_EXPAND_CACHE_VERSION = 13
PRACTICE_FAST_INTENT_CONTRACT_VERSION = "expansion_intent_v2"
# An expansion can be nearly one segment-cache lifetime old when it discovers a
# newly analyzed source. Keeping it for two lifetimes guarantees that source's
# subsequent valid segment-cache lifetime never triggers another expansion call.
PRACTICE_FAST_EXPAND_CACHE_TTL_SEC = 2 * SEGMENT_CACHE_TTL_SEC
RECOVERY_REASON_ZERO_SEARCH_RESULTS = "zero_search_results"
RECOVERY_REASON_ZERO_VALID_CLIPS = "zero_valid_clips"


def _normalized_recovery_reason(value: object) -> str:
    return (
        RECOVERY_REASON_ZERO_VALID_CLIPS
        if str(value or "").strip().casefold()
        == RECOVERY_REASON_ZERO_VALID_CLIPS
        else RECOVERY_REASON_ZERO_SEARCH_RESULTS
    )


def _normalized_rejected_video_ids(values: Sequence[str]) -> list[str]:
    return sorted({
        video_id
        for raw_value in values
        if (video_id := " ".join(str(raw_value or "").split()))
    })


def _require_source_occurrence_schema(schema: dict[str, object]) -> None:
    """Keep old fixtures readable while requiring live intent identity fields."""
    required = schema.setdefault("required", [])
    if isinstance(required, list):
        for field_name in ("source_occurrence", "relationship_topology"):
            if field_name not in required:
                required.append(field_name)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field_name in ("source_occurrence", "relationship_topology"):
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict):
                field_schema.pop("default", None)


def _require_joint_structures_schema(schema: dict[str, object]) -> None:
    """Require the relationship graph in Gemini output, including an empty list."""
    required = schema.setdefault("required", [])
    if isinstance(required, list) and "joint_structures" not in required:
        required.append("joint_structures")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        field_schema = properties.get("joint_structures")
        if isinstance(field_schema, dict):
            field_schema.pop("default", None)


class _PracticeFastIntentConstraint(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=_require_source_occurrence_schema,
    )

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
    source_occurrence: int = Field(default=0, ge=0, strict=True)
    requirement: str = Field(min_length=1, max_length=240)
    relationship_topology: Literal[
        "directed",
        "reciprocal",
        "symmetric",
        "ordered",
        "unspecified",
        "not_applicable",
    ] = "not_applicable"

class _PracticeFastJointStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member_constraint_ids: list[str] = Field(min_length=2, max_length=16)
    relation_constraint_id: str = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _unique_members(self):
        if len(self.member_constraint_ids) != len(set(self.member_constraint_ids)):
            raise ValueError("joint member constraint ids must be unique")
        if self.relation_constraint_id in self.member_constraint_ids:
            raise ValueError("joint relation id cannot also be a member id")
        return self


class _PracticeFastQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=240)
    preserved_constraint_ids: list[str] = Field(min_length=1, max_length=16)


class _PracticeFastExpansion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra=_require_joint_structures_schema,
    )

    corrected: str = Field(
        min_length=1,
        max_length=220,
        description=(
            "Concise standalone learning-intent summary preserving the subject, "
            "level, requested facets, relationships, tasks, outcomes, and order."
        ),
    )
    intent_constraints: list[_PracticeFastIntentConstraint] = Field(
        min_length=1,
        max_length=16,
    )
    joint_structures: list[_PracticeFastJointStructure] = Field(
        default_factory=list,
        max_length=8,
    )
    summary_preserved_constraint_ids: list[str] = Field(
        min_length=1,
        max_length=16,
        description=(
            "Every intent constraint ID preserved by corrected, each listed "
            "exactly once."
        ),
    )
    queries: list[_PracticeFastQuery]

    @model_validator(mode="after")
    def _unique_intent_ids(self):
        if any(
            (
                constraint.kind == "relationship"
                and constraint.relationship_topology == "not_applicable"
            )
            or (
                constraint.kind != "relationship"
                and constraint.relationship_topology != "not_applicable"
            )
            for constraint in self.intent_constraints
        ):
            raise ValueError(
                "relationship topology must match the relationship constraint kind"
            )
        ids = [constraint.constraint_id for constraint in self.intent_constraints]
        if len(set(ids)) != len(ids):
            raise ValueError("intent constraint ids must be unique")
        id_set = set(ids)
        if any(
            structure.relation_constraint_id not in id_set
            or any(
                member_id not in id_set
                for member_id in structure.member_constraint_ids
            )
            for structure in self.joint_structures
        ):
            raise ValueError("joint structures must reference declared constraint ids")
        structure_keys = [
            (
                tuple(structure.member_constraint_ids),
                structure.relation_constraint_id,
            )
            for structure in self.joint_structures
        ]
        if len(structure_keys) != len(set(structure_keys)):
            raise ValueError("joint structures must be unique")
        return self


class _PracticeFastRequestIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_request: str = Field(min_length=1)
    constraints: list[_PracticeFastIntentConstraint] = Field(
        min_length=1,
        max_length=16,
    )
    joint_structures: list[_PracticeFastJointStructure] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def _valid_references(self):
        ids = [constraint.constraint_id for constraint in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("intent constraint ids must be unique")
        id_set = set(ids)
        if any(
            structure.relation_constraint_id not in id_set
            or any(
                member_id not in id_set
                for member_id in structure.member_constraint_ids
            )
            for structure in self.joint_structures
        ):
            raise ValueError("joint structures must reference declared constraint ids")
        structure_keys = [
            (
                tuple(structure.member_constraint_ids),
                structure.relation_constraint_id,
            )
            for structure in self.joint_structures
        ]
        if len(structure_keys) != len(set(structure_keys)):
            raise ValueError("joint structures must be unique")
        return self


class _PracticeFastIntentContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["expansion_intent_v2"]
    request_intent: _PracticeFastRequestIntent


def _validated_selector_intent_contract(
    raw_contract: object,
) -> dict[str, object] | None:
    """Use the clip selector's exact structural trust gate as the authority."""
    if not isinstance(raw_contract, dict):
        return None
    from ...pipeline.gemini_segment import _trusted_request_intent_from_settings

    trusted = _trusted_request_intent_from_settings({
        "_segment_intent_contract": raw_contract,
    })
    if trusted is None:
        return None
    return {
        "version": PRACTICE_FAST_INTENT_CONTRACT_VERSION,
        "request_intent": trusted.model_dump(mode="json"),
    }


_PRACTICE_FAST_SYSTEM = """You compress a user's learning request and expand it into a diverse set of
YouTube search queries that maximize topical coverage.

Do this:
1. Spellcheck the input and compress it into corrected: one concise, standalone,
   intent-preserving learning summary. Preserve the governing subject, learner level, requested
   concepts, relationships, tasks, outcomes, and teaching order. Remove conversational filler,
   but never broaden, add, or drop a requirement. corrected must make sense without the original
   request, must not merely copy or expand a long conversational sentence, and must be at most
   200 characters. Use compact wording rather than dropping a constraint to meet the limit.
2. Infer the most likely intent or sense.
3. From corrected, produce up to N concise queries that a person would actually search on YouTube.
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
   that introduced each constraint into source_phrase and return its zero-based
   source_occurrence in the exact request. Use zero unless the identical phrase appears earlier.
   Do not collapse a requested task or facet
   into the subject. SUBJECT means the governing named topic, law, concept, or object; components
   and named list members under that governing topic are SCOPE constraints. For
   "Explain Newton's second law F=ma with net force, mass, acceleration, units, and solving for
   each variable," Newton's second law F=ma is SUBJECT,
   Explain is TASK, net force, mass, acceleration, and units are four separate SCOPE constraints,
   and solving for each variable is OUTCOME.
7. Every intent constraint sets relationship_topology. Use not_applicable for every
   non-relationship constraint. For a relationship constraint use reciprocal for explicit two-way
   interactions such as action-reaction, symmetric for balance/equivalence/undirected comparison,
   directed for one-way causal or dependency relationships, ordered for explicit stage/curriculum
   sequences, and unspecified only when the request does not determine a direction class. Return
   joint_structures=[] when no explicit comparison, contrast, transition, or ordered relationship
   is requested. Otherwise return one item per relationship: member_constraint_ids names its
   separately declared sides or stages and relation_constraint_id names its separately declared
   relationship constraint. Never bundle named members merely to make this graph.
8. For corrected, return summary_preserved_constraint_ids containing exactly every intent
   constraint ID that corrected preserves. corrected must preserve all of them. List each ID
   exactly once, with no omissions, duplicates, or unknown IDs; if any requirement is absent,
   revise corrected before returning it.
9. The first broad query must preserve every constraint. Every later focused query must preserve
   every subject constraint plus one or more distinct task, relationship, scope, format, or outcome
   constraints. Collectively target every named facet or list member; when the request says each,
   every, or all, give distinct members focused coverage where N permits.
10. For every query, return exactly the IDs of the constraints it preserves. Synonyms and natural
   YouTube wording are welcome, but focused queries must not lose the governing subject or drift
   into an adjacent field.

Return only the requested JSON object with corrected, intent_constraints, joint_structures,
summary_preserved_constraint_ids, and queries. corrected is the downstream learning intent; keep
it separate from the retrieval queries. Return N optimized search queries; do not automatically
spend one query on the raw literal wording."""


def literal_fallback(topic: str, n: int) -> dict:
    literal = " ".join(str(topic or "").split())
    return {
        "corrected": literal,
        "queries": [literal] if literal and int(n) > 0 else [],
        "provider_used": "literal_fallback",
    }


def _expansion_cache_key(
    topic: str,
    n: int,
    level: str | None,
    *,
    tried_queries: Sequence[str] = (),
    recovery_reason: str | None = None,
    rejected_video_ids: Sequence[str] = (),
) -> str:
    contract = {
        "version": PRACTICE_FAST_EXPAND_CACHE_VERSION,
        "topic": topic,
        "count": int(n),
        "level": " ".join(str(level or "").split()),
        "model": PRACTICE_FAST_EXPAND_MODEL,
        "prompt_sha256": hashlib.sha256(_PRACTICE_FAST_SYSTEM.encode("utf-8")).hexdigest(),
        "schema": _PracticeFastExpansion.model_json_schema(),
    }
    normalized_tried = _normalize(list(tried_queries), len(tried_queries))
    if normalized_tried:
        # A recovery must not reuse the healthy-path expansion or another
        # recovery whose search/clip evidence rejected different inputs.
        contract["zero_result_recovery"] = True
        contract["recovery_reason"] = _normalized_recovery_reason(
            recovery_reason
        )
        contract["tried_queries"] = normalized_tried
        normalized_rejected_ids = _normalized_rejected_video_ids(
            rejected_video_ids
        )
        if normalized_rejected_ids:
            contract["rejected_video_ids"] = normalized_rejected_ids
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
    trusted_contract = _validated_selector_intent_contract(
        payload.get("intent_contract")
    )
    if trusted_contract is None:
        return None
    try:
        intent_contract = _PracticeFastIntentContract.model_validate(
            trusted_contract
        ).model_dump(mode="json")
    except (TypeError, ValueError):
        return None
    return {
        "corrected": corrected,
        "queries": queries,
        "provider_used": "gemini",
        "intent_contract": intent_contract,
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
                            "intent_contract": result["intent_contract"],
                        }
                    ),
                    "created_at": now_iso(),
                },
                pk="cache_key",
            )
    except Exception as exc:
        logger.debug("Practice expansion cache write unavailable: %s", exc)


def _key(value: object) -> str:
    return semantic_key(value)


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
    return list(semantic_tokens(value, preserve_terminal_suffix=True))


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
    """Prefer broad-plus-focused subject-grounded retrieval branches."""
    constraints = list(parsed.intent_constraints)
    constraint_ids = {constraint.constraint_id for constraint in constraints}
    if not constraint_ids:
        return []
    summary_ids = list(parsed.summary_preserved_constraint_ids)
    if len(summary_ids) != len(set(summary_ids)):
        return []
    if set(summary_ids) != constraint_ids:
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
    primary_broad = normalized_model_broad[0] if normalized_model_broad else ""
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
        return _normalize(normalized_model_broad, count)
    if len(non_subject_ids) <= 2:
        normalized_focused.extend(
            (query, frozenset(non_subject_ids))
            for query in normalized_model_broad[1:]
        )
    if count <= 1:
        return _normalize(
            [primary_broad or normalized_focused[0][0]]
            if primary_broad or normalized_focused
            else [],
            count,
        )
    if not primary_broad and not normalized_focused:
        return []

    focused_limit = max(0, count - int(bool(primary_broad)))
    selected_indices = _covering_focus_indices(
        normalized_focused,
        non_subject_ids,
        focused_limit,
    )
    selected_index_set = set(selected_indices or ())
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
            *([primary_broad] if primary_broad else []),
            *(query for query, _signature in selected_focused),
            *normalized_model_broad[1:],
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
    tried_queries: Sequence[str] = (),
    recovery_reason: str | None = None,
    rejected_video_ids: Sequence[str] = (),
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
        "Return corrected, intent_constraints, joint_structures, "
        "summary_preserved_constraint_ids, and queries as JSON."
    )
    normalized_tried = _normalize(list(tried_queries), len(tried_queries))
    if normalized_tried:
        normalized_reason = _normalized_recovery_reason(recovery_reason)
        if normalized_reason == RECOVERY_REASON_ZERO_VALID_CLIPS:
            normalized_rejected_ids = _normalized_rejected_video_ids(
                rejected_video_ids
            )
            user += (
                "\nZERO_VALID_CLIP_RECOVERY: YouTube videos were found, but "
                "transcript retrieval, selector/audit validation, or boundary "
                "validation produced zero valid educational clips. Return N "
                "novel, concise, intent-grounded alternatives aimed at different "
                "sources; do not repeat the failed search wording or rejected "
                "videos.\nPreviously tried queries: "
                + json.dumps(normalized_tried, ensure_ascii=True)
                + "\nRejected video IDs: "
                + json.dumps(normalized_rejected_ids, ensure_ascii=True)
            )
        else:
            user += (
                "\nZERO_SEARCH_RESULT_RECOVERY: Supadata returned zero eligible "
                "YouTube videos for every previously tried query. Return N novel, "
                "concise, intent-grounded alternatives and do not repeat or "
                "paraphrase only the same failed search wording.\nPreviously "
                "tried queries: "
                + json.dumps(normalized_tried, ensure_ascii=True)
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
    tried_queries: Sequence[str] = (),
    recovery_reason: str | None = None,
    rejected_video_ids: Sequence[str] = (),
) -> str:
    return run_cancellable(
        lambda: _practice_fast_gemini_raw_async(
            topic,
            n,
            model=model,
            level=level,
            should_cancel=should_cancel,
            context=context,
            tried_queries=tried_queries,
            recovery_reason=recovery_reason,
            rejected_video_ids=rejected_video_ids,
        ),
        should_cancel,
    )


def _practice_fast_failure_is_retryable(exc: Exception) -> bool:
    """Retry only transport/local-contract failures, 408/429, and 5xx."""
    error_types = {base.__name__ for base in type(exc).__mro__}
    if (
        isinstance(exc, (CancellationError, ProviderConfigurationError))
        or "GeminiCancelledError" in error_types
        or "GeminiBlockedResponseError" in error_types
    ):
        return False

    telemetry = getattr(exc, "telemetry", None)
    response = getattr(exc, "response", None)
    raw_statuses = (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(response, "status_code", None),
        (
            telemetry.get("provider_status_code")
            if isinstance(telemetry, dict)
            else getattr(telemetry, "provider_status_code", None)
        ),
    )
    status = None
    for raw_status in raw_statuses:
        raw_status = getattr(raw_status, "value", raw_status)
        try:
            status = int(raw_status) if raw_status is not None else None
        except (TypeError, ValueError, OverflowError):
            continue
        if status is not None:
            break
    if status is not None:
        return status in {408, 429} or 500 <= status <= 599

    retryable = getattr(exc, "retryable", None)
    if retryable is None and telemetry is not None:
        retryable = (
            telemetry.get("retryable")
            if isinstance(telemetry, dict)
            else getattr(telemetry, "retryable", None)
        )
    # Statusless transport and local response-contract failures are safe to
    # retry once. Typed permanent provider failures expose retryable=False.
    return True if retryable is None else bool(retryable)


def expand_query_practice_fast(
    topic: str,
    n: int,
    level: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    context: "GenerationContext | None" = None,
    tried_queries: Sequence[str] = (),
    recovery_reason: str | None = None,
    rejected_video_ids: Sequence[str] = (),
) -> dict:
    """Return cached Flash search terms, failing safely to the literal request.

    Expansion deliberately does not reserve the Supadata search budget tracked by
    ``context``. The context enables durable cache reuse and usage accounting.
    """
    topic = " ".join(str(topic or "").split())
    count = max(0, int(n))
    if not topic or count == 0:
        return literal_fallback(topic, count)
    normalized_tried = _normalize(list(tried_queries), len(tried_queries))
    tried_keys = {_key(query) for query in normalized_tried}
    recovering_zero_results = bool(normalized_tried)
    raise_if_cancelled(should_cancel)
    cache_key = (
        _expansion_cache_key(
            topic,
            count,
            level,
            tried_queries=normalized_tried,
            recovery_reason=recovery_reason,
            rejected_video_ids=rejected_video_ids,
        )
        if context is not None
        else ""
    )
    cached = _read_cached_expansion(cache_key, count) if cache_key else None
    if cached is not None:
        if recovering_zero_results:
            cached = {
                **cached,
                "queries": [
                    query
                    for query in cached["queries"]
                    if _key(query) not in tried_keys
                ],
            }
            if not cached["queries"]:
                cached = None
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
            if recovering_zero_results:
                cached = {
                    **cached,
                    "queries": [
                        query
                        for query in cached["queries"]
                        if _key(query) not in tried_keys
                    ],
                }
                if not cached["queries"]:
                    cached = None
        if cached is not None:
            context.increment_counter("expansion_cache_hits")
            context.record_cache_hit(
                provider="gemini",
                operation="expansion",
                metadata={"cache_key": cache_key},
            )
            return cached
        for attempt in range(PRACTICE_FAST_EXPAND_ATTEMPTS):
            raise_if_cancelled(should_cancel)
            try:
                kwargs = {
                    "model": PRACTICE_FAST_EXPAND_MODEL,
                    "level": level,
                    "should_cancel": should_cancel,
                }
                if context is not None:
                    kwargs["context"] = context
                if recovering_zero_results:
                    kwargs["tried_queries"] = normalized_tried
                    kwargs["recovery_reason"] = _normalized_recovery_reason(
                        recovery_reason
                    )
                    kwargs["rejected_video_ids"] = (
                        _normalized_rejected_video_ids(rejected_video_ids)
                    )
                raw = _practice_fast_gemini_raw(topic, count, **kwargs)
                parsed = _PracticeFastExpansion.model_validate_json(raw)
                corrected = " ".join(str(parsed.corrected or topic).split()) or topic
                queries = _validated_ai_queries(topic, parsed, count)
                if recovering_zero_results:
                    queries = [
                        query
                        for query in queries
                        if _key(query) not in tried_keys
                    ]
                if not queries:
                    raise ValueError(
                        "Gemini returned no novel search query preserving the exact "
                        "intent contract"
                        if recovering_zero_results
                        else "Gemini returned no search query preserving the exact "
                        "intent contract"
                    )
                raise_if_cancelled(should_cancel)
                intent_contract_model = _PracticeFastIntentContract(
                    version=PRACTICE_FAST_INTENT_CONTRACT_VERSION,
                    request_intent=_PracticeFastRequestIntent(
                        exact_request=topic,
                        constraints=list(parsed.intent_constraints),
                        joint_structures=list(parsed.joint_structures),
                    ),
                )
                intent_contract = _validated_selector_intent_contract(
                    intent_contract_model.model_dump(
                        mode="json",
                        exclude_unset=True,
                    )
                )
                if intent_contract is None:
                    raise ValueError(
                        "Gemini returned an intent contract rejected by the "
                        "clip selector"
                    )
                result = {
                    "corrected": corrected,
                    "queries": queries,
                    "provider_used": "gemini",
                    "intent_contract": intent_contract,
                }
                if cache_key:
                    _write_cached_expansion(cache_key, result)
                return result
            except CancellationError:
                raise
            except Exception as exc:
                raise_if_cancelled(should_cancel)
                errors.append(
                    f"{PRACTICE_FAST_EXPAND_MODEL} attempt {attempt + 1}: {exc}"
                )
                if (
                    attempt + 1 >= PRACTICE_FAST_EXPAND_ATTEMPTS
                    or not _practice_fast_failure_is_retryable(exc)
                ):
                    break
    if recovering_zero_results:
        logger.info(
            "practice-fast Gemini zero-result recovery unavailable: %s",
            "; ".join(errors),
        )
        return {
            "corrected": topic,
            "queries": [],
            "provider_used": "gemini_recovery_unavailable",
        }
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
