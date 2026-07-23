"""Query expansion for the existing planner and the opt-in practice-fast path.

Topic correction, aliases, and semantic expansion are owned by the persisted
TopicExpansionService. The existing ``expand_query`` API retains its stable templates.
Production retrieval uses ``expand_query_practice_fast`` for a cached Gemini plan.
Recoverable failures receive bounded corrective Gemini retries; exhausted failures
remain typed so the durable generation controller can retry them.
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

from ... import gemini_client
from ...concept_tokens import semantic_key, semantic_tokens
from ...intent_obligations import intent_obligation_key
from . import config
from .cancellation import raise_if_cancelled, run_cancellable
from .errors import (
    CancellationError,
    ModelUnavailableError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderTransientError,
)
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
PRACTICE_FAST_EXPAND_FALLBACK_MODEL = "gemini-3.6-flash"
# Gemini rejects manually configured deadlines below ten seconds.
PRACTICE_FAST_EXPAND_TIMEOUT_MS = 10_000
PRACTICE_FAST_EXPAND_OUTPUT_TOKENS = 4_096
PRACTICE_FAST_EXPAND_FALLBACK_OUTPUT_TOKENS = 4_096
PRACTICE_FAST_EXPAND_ATTEMPTS = 3
PRACTICE_FAST_EXPAND_CACHE_VERSION = 15
PRACTICE_FAST_INTENT_CONTRACT_VERSION = "expansion_intent_v2"
# An expansion can be nearly one segment-cache lifetime old when it discovers a
# newly analyzed source. Keeping it for two lifetimes guarantees that source's
# subsequent valid segment-cache lifetime never triggers another expansion call.
PRACTICE_FAST_EXPAND_CACHE_TTL_SEC = 2 * SEGMENT_CACHE_TTL_SEC
RECOVERY_REASON_ZERO_SEARCH_RESULTS = "zero_search_results"
RECOVERY_REASON_ZERO_VALID_CLIPS = "zero_valid_clips"
PRACTICE_FAST_SOURCE_CONTEXT_MAX_CHARS = 600


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


def _normalized_source_context(value: object) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= PRACTICE_FAST_SOURCE_CONTEXT_MAX_CHARS:
        return normalized
    return (
        normalized[:PRACTICE_FAST_SOURCE_CONTEXT_MAX_CHARS]
        .rsplit(" ", 1)[0]
        .strip()
    )


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

    member_constraint_ids: list[str] = Field(
        min_length=2,
        max_length=16,
        description=(
            "At least two separately declared side or stage constraint IDs. "
            "Never include relation_constraint_id."
        ),
    )
    relation_constraint_id: str = Field(
        min_length=1,
        max_length=32,
        description=(
            "The separately declared relationship constraint ID; it is not a "
            "member_constraint_id."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _remove_redundant_relation_member(cls, value):
        if not isinstance(value, dict):
            return value
        relation_id = value.get("relation_constraint_id")
        member_ids = value.get("member_constraint_ids")
        if not relation_id or not isinstance(member_ids, list):
            return value
        if relation_id not in member_ids:
            return value
        normalized = dict(value)
        normalized_members = [
            member_id for member_id in member_ids if member_id != relation_id
        ]
        if len(normalized_members) < 2:
            raise ValueError(
                "joint member_constraint_ids must contain at least two "
                "non-relation side or stage constraint IDs; "
                "relation_constraint_id must not also be a member"
            )
        normalized["member_constraint_ids"] = normalized_members
        return normalized

    @model_validator(mode="after")
    def _unique_members(self):
        if len(self.member_constraint_ids) != len(set(self.member_constraint_ids)):
            raise ValueError("joint member constraint ids must be unique")
        return self


class _PracticeFastCoordinatedGroup(BaseModel):
    """Gemini's explicit audit that coordinated members stayed atomic."""

    model_config = ConfigDict(extra="forbid")

    member_constraint_ids: list[str] = Field(
        min_length=2,
        max_length=16,
        description=(
            "Distinct atomic constraint IDs for every explicitly coordinated "
            "task, facet, list member, side, or stage."
        ),
    )

    @model_validator(mode="after")
    def _unique_members(self):
        if len(self.member_constraint_ids) != len(set(self.member_constraint_ids)):
            raise ValueError("coordinated group member ids must be unique")
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
    reverse_coverage_complete: Literal[True] = Field(
        description=(
            "True only after independently scanning the exact request from end "
            "to beginning and mapping every explicit requirement."
        ),
    )
    reverse_coverage_constraint_ids: list[str] = Field(
        min_length=1,
        max_length=16,
        description=(
            "Every atomic constraint found by the reverse request audit, each "
            "listed exactly once."
        ),
    )
    acquisition_obligation_constraint_ids: list[str] = Field(
        min_length=1,
        max_length=16,
        description=(
            "Every independently teachable request obligation acquisition "
            "should seek. Exclude contextual level or style constraints that a "
            "clip objective cannot fulfill."
        ),
    )
    coordinated_groups: list[_PracticeFastCoordinatedGroup] = Field(
        max_length=8,
        description=(
            "One group for each explicitly coordinated set; empty only when the "
            "request has no coordinated tasks, facets, list members, sides, or stages."
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
        reverse_ids = list(self.reverse_coverage_constraint_ids)
        if (
            len(reverse_ids) != len(set(reverse_ids))
            or set(reverse_ids) != id_set
        ):
            raise ValueError(
                "reverse coverage must reference every intent constraint "
                "exactly once"
            )
        acquisition_ids = list(self.acquisition_obligation_constraint_ids)
        if (
            len(acquisition_ids) != len(set(acquisition_ids))
            or not set(acquisition_ids).issubset(id_set)
        ):
            raise ValueError(
                "acquisition obligations must be unique declared intent "
                "constraint ids"
            )
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
        coordinated_keys: list[tuple[str, ...]] = []
        for group in self.coordinated_groups:
            member_ids = tuple(group.member_constraint_ids)
            if (
                any(member_id not in id_set for member_id in member_ids)
                or any(
                    next(
                        constraint
                        for constraint in self.intent_constraints
                        if constraint.constraint_id == member_id
                    ).kind
                    == "relationship"
                    for member_id in member_ids
                )
            ):
                raise ValueError(
                    "coordinated groups must reference declared atomic "
                    "non-relationship constraints"
                )
            coordinated_keys.append(member_ids)
        if len(coordinated_keys) != len(set(coordinated_keys)):
            raise ValueError("coordinated groups must be unique")
        if any(
            tuple(structure.member_constraint_ids) not in coordinated_keys
            for structure in self.joint_structures
        ):
            raise ValueError(
                "every joint structure must retain its atomic members in a "
                "coordinated group"
            )
        required_acquisition_ids = {
            constraint.constraint_id
            for constraint in self.intent_constraints
            if constraint.kind in {
                "subject",
                "task",
                "relationship",
                "outcome",
            }
        }
        required_acquisition_ids.update(
            member_id
            for group in self.coordinated_groups
            for member_id in group.member_constraint_ids
        )
        required_acquisition_ids.update(
            structure.relation_constraint_id
            for structure in self.joint_structures
        )
        if not required_acquisition_ids.issubset(set(acquisition_ids)):
            raise ValueError(
                "acquisition obligations must retain every teachable task, "
                "relationship, outcome, subject, and coordinated member"
            )
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


def trusted_intent_obligation_keys(
    raw_contract: object,
    constraint_ids: Sequence[str] | None = None,
) -> set[str]:
    """Derive durable obligation identities from a selector-trusted contract."""
    keys_by_constraint = trusted_intent_obligation_keys_by_constraint(
        raw_contract
    )
    if not keys_by_constraint:
        return set()
    if constraint_ids is None:
        return set(keys_by_constraint.values())
    normalized_ids = [
        " ".join(str(value or "").split())
        for value in constraint_ids
        if " ".join(str(value or "").split())
    ]
    if (
        len(normalized_ids) != len(set(normalized_ids))
        or not set(normalized_ids).issubset(keys_by_constraint)
    ):
        return set()
    trusted = _trusted_request_intent(raw_contract)
    if trusted is None:
        return set()
    required_ids = {
        str(constraint.constraint_id)
        for constraint in trusted.constraints
        if constraint.kind in {
            "subject",
            "task",
            "relationship",
            "outcome",
        }
    }
    if not required_ids.issubset(normalized_ids):
        return set()
    return {
        keys_by_constraint[constraint_id]
        for constraint_id in normalized_ids
    }


def _trusted_request_intent(raw_contract: object):
    if not isinstance(raw_contract, dict):
        return None
    from ...pipeline.gemini_segment import _trusted_request_intent_from_settings

    return _trusted_request_intent_from_settings({
        "_segment_intent_contract": raw_contract,
    })


def trusted_intent_obligation_keys_by_constraint(
    raw_contract: object,
) -> dict[str, str]:
    """Map each selector-trusted constraint ID to its durable obligation key."""
    trusted = _trusted_request_intent(raw_contract)
    if trusted is None:
        return {}
    from ...pipeline.gemini_segment import _canonical_intent_constraint_sources

    canonical_sources = _canonical_intent_constraint_sources(
        trusted.exact_request,
        trusted.constraints,
    )
    return {
        str(constraint.constraint_id): key
        for constraint in trusted.constraints
        if (
            key := intent_obligation_key(
                *canonical_sources[str(constraint.constraint_id)]
            )
        )
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
9. Independently re-read the exact request from its final words back to its beginning. Do not use
   the forward intent_constraints list as the source of this audit. Find every explicit task,
   facet, named list member, relationship, requested outcome, format, scope qualifier, and teaching
   order. Put the corresponding atomic constraint IDs in reverse_coverage_constraint_ids exactly
   once and set reverse_coverage_complete=true only after every discovered requirement maps to one.
   If the reverse audit finds an omission or a combined constraint, repair intent_constraints first.
10. For every explicit coordination of two or more tasks, facets, named list members, comparison
   sides, or ordered stages, return one coordinated_groups item containing a distinct constraint ID
   for every member. Never use one umbrella member for a coordinated list. Include the member group
   used by every joint_structure; return coordinated_groups=[] only when there is no explicit
   coordination.
11. Return acquisition_obligation_constraint_ids for every independently teachable request
   obligation that source acquisition should seek. It must include every subject, task,
   relationship, outcome, and coordinated_groups member. Include a scope or format when a clip can
   substantively fulfill it (for example a named facet, unit explanation, comparison side, worked
   example, or requested problem type). Exclude purely contextual viewer level, pacing, tone, or
   source-format preferences that no clip objective can fulfill.
12. The first broad query must preserve every constraint. Every later focused query must preserve
   every subject constraint plus one or more distinct task, relationship, scope, format, or outcome
   constraints. Collectively target every named facet or list member; when the request says each,
   every, or all, give distinct members focused coverage where N permits.
13. For every query, return exactly the IDs of the constraints it preserves. Synonyms and natural
   YouTube wording are welcome, but focused queries must not lose the governing subject or drift
   into an adjacent field.

Return only the requested JSON object with corrected, intent_constraints, joint_structures,
summary_preserved_constraint_ids, reverse_coverage_complete,
reverse_coverage_constraint_ids, acquisition_obligation_constraint_ids,
coordinated_groups, and queries. corrected is the downstream learning intent; keep it separate
from the retrieval queries. Return N optimized search queries; do not automatically spend one
query on the raw literal wording."""


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
    source_context: str | None = None,
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
    normalized_source_context = _normalized_source_context(source_context)
    if normalized_source_context:
        contract["source_context"] = normalized_source_context
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
    raw_acquisition_ids = payload.get("acquisition_obligation_constraint_ids")
    if not isinstance(raw_acquisition_ids, list):
        return None
    acquisition_ids = [
        " ".join(str(value or "").split())
        for value in raw_acquisition_ids
        if " ".join(str(value or "").split())
    ]
    if (
        len(acquisition_ids) != len(raw_acquisition_ids)
        or not trusted_intent_obligation_keys(
            intent_contract,
            acquisition_ids,
        )
    ):
        return None
    query_metadata = _validated_query_metadata(
        payload.get("query_metadata"),
        queries=queries,
        intent_contract=intent_contract,
        acquisition_constraint_ids=acquisition_ids,
    )
    if query_metadata is None:
        return None
    return {
        "corrected": corrected,
        "queries": queries,
        "provider_used": "gemini",
        "intent_contract": intent_contract,
        "acquisition_obligation_constraint_ids": acquisition_ids,
        "query_metadata": query_metadata,
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
                            "acquisition_obligation_constraint_ids": result[
                                "acquisition_obligation_constraint_ids"
                            ],
                            "query_metadata": result["query_metadata"],
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


def _validated_query_metadata(
    raw_metadata: object,
    *,
    queries: Sequence[str],
    intent_contract: object,
    acquisition_constraint_ids: Sequence[str],
) -> list[dict[str, object]] | None:
    """Validate cached/live query provenance and derive trusted objective keys."""
    if not isinstance(raw_metadata, list):
        return None
    keys_by_constraint = trusted_intent_obligation_keys_by_constraint(
        intent_contract
    )
    if not keys_by_constraint:
        return None
    acquisition_ids = {
        " ".join(str(value or "").split())
        for value in acquisition_constraint_ids
        if " ".join(str(value or "").split())
    }
    if not acquisition_ids or not acquisition_ids.issubset(keys_by_constraint):
        return None
    trusted = _trusted_request_intent(intent_contract)
    if trusted is None:
        return None
    subject_ids = {
        str(constraint.constraint_id)
        for constraint in trusted.constraints
        if constraint.kind == "subject"
    }
    by_query_key: dict[str, dict[str, object]] = {}
    for raw_item in raw_metadata:
        if not isinstance(raw_item, dict):
            return None
        text = " ".join(str(raw_item.get("text") or "").split())
        query_key = _key(text)
        raw_ids = raw_item.get("preserved_constraint_ids")
        if not text or not query_key or not isinstance(raw_ids, list):
            return None
        constraint_ids = [
            " ".join(str(value or "").split())
            for value in raw_ids
            if " ".join(str(value or "").split())
        ]
        if (
            len(constraint_ids) != len(raw_ids)
            or len(constraint_ids) != len(set(constraint_ids))
            or not set(constraint_ids).issubset(keys_by_constraint)
            or not subject_ids.issubset(constraint_ids)
            or query_key in by_query_key
        ):
            return None
        objective_keys = sorted({
            keys_by_constraint[constraint_id]
            for constraint_id in constraint_ids
            if constraint_id in acquisition_ids
        })
        if not objective_keys:
            return None
        raw_objective_keys = raw_item.get("intent_obligation_keys")
        if raw_objective_keys is not None and (
            not isinstance(raw_objective_keys, list)
            or [
                " ".join(str(value or "").split())
                for value in raw_objective_keys
                if " ".join(str(value or "").split())
            ] != objective_keys
        ):
            return None
        covers_all_intent_constraints = set(
            keys_by_constraint
        ).issubset(constraint_ids)
        focused_objective_keys = (
            []
            if covers_all_intent_constraints
            else sorted({
                keys_by_constraint[constraint_id]
                for constraint_id in constraint_ids
                if (
                    constraint_id in acquisition_ids
                    and constraint_id not in subject_ids
                )
            })
        )
        raw_focused_keys = raw_item.get(
            "focused_intent_obligation_keys"
        )
        if raw_focused_keys is not None and (
            not isinstance(raw_focused_keys, list)
            or [
                " ".join(str(value or "").split())
                for value in raw_focused_keys
                if " ".join(str(value or "").split())
            ] != focused_objective_keys
        ):
            return None
        by_query_key[query_key] = {
            "text": text,
            "preserved_constraint_ids": constraint_ids,
            "intent_obligation_keys": objective_keys,
            "focused_intent_obligation_keys": focused_objective_keys,
            "covers_all_intent_constraints": covers_all_intent_constraints,
        }
    ordered: list[dict[str, object]] = []
    for query in queries:
        item = by_query_key.get(_key(query))
        if item is None:
            return None
        ordered.append({**item, "text": " ".join(str(query).split())})
    return ordered


def _without_tried_expansion_queries(
    expansion: dict,
    tried_keys: set[str],
) -> dict:
    queries = [
        query
        for query in expansion.get("queries") or []
        if _key(query) not in tried_keys
    ]
    allowed = {_key(query) for query in queries}
    return {
        **expansion,
        "queries": queries,
        "query_metadata": [
            item
            for item in expansion.get("query_metadata") or []
            if isinstance(item, dict)
            and _key(item.get("text")) in allowed
        ],
    }


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
    *,
    query_metadata_out: list[dict[str, object]] | None = None,
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
    constraint_ids_by_query_key: dict[str, list[str]] = {}
    non_subject_ids = constraint_ids - subject_ids
    for query in parsed.queries:
        preserved_ids = [
            " ".join(str(value or "").split())
            for value in query.preserved_constraint_ids
            if " ".join(str(value or "").split())
        ]
        preserved = set(preserved_ids)
        if (
            not preserved
            or len(preserved_ids) != len(preserved)
            or not preserved.issubset(constraint_ids)
            or not subject_ids.issubset(preserved)
        ):
            continue
        if preserved == constraint_ids:
            constraint_ids_by_query_key[_key(query.text)] = preserved_ids
            broad.append(query.text)
            continue
        constraint_ids_by_query_key.setdefault(_key(query.text), preserved_ids)
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

    def finalize(candidates: list[str]) -> list[str]:
        queries = _normalize(candidates, count)
        if query_metadata_out is not None:
            query_metadata_out.extend(
                {
                    "text": query,
                    "preserved_constraint_ids": list(
                        constraint_ids_by_query_key.get(_key(query), ())
                    ),
                }
                for query in queries
            )
        return queries

    if not non_subject_ids:
        return finalize(normalized_model_broad)
    if len(non_subject_ids) <= 2:
        normalized_focused.extend(
            (query, frozenset(non_subject_ids))
            for query in normalized_model_broad[1:]
        )
    if count <= 1:
        return finalize(
            [primary_broad or normalized_focused[0][0]]
            if primary_broad or normalized_focused
            else [],
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
    return finalize(
        [
            *([primary_broad] if primary_broad else []),
            *(query for query, _signature in selected_focused),
            *normalized_model_broad[1:],
        ],
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
    source_context: str | None = None,
    tried_queries: Sequence[str] = (),
    recovery_reason: str | None = None,
    rejected_video_ids: Sequence[str] = (),
    validation_feedback: str | None = None,
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
        "summary_preserved_constraint_ids, reverse_coverage_complete, "
        "reverse_coverage_constraint_ids, "
        "acquisition_obligation_constraint_ids, coordinated_groups, and "
        "queries as JSON."
    )
    normalized_source_context = _normalized_source_context(source_context)
    if normalized_source_context:
        user += (
            "\nSOURCE_GROUNDED_CONTEXT (backend supplied): "
            + json.dumps(normalized_source_context, ensure_ascii=True)
            + "\nUse this context to disambiguate the User topic and make "
            "corrected a concise, concrete learning intent that carries its "
            "relevant process and learning outcome into downstream clip "
            "selection. It is not a second user request: do not create intent "
            "constraints, source_occurrence text, or mandatory scope from "
            "context alone. Every constraint and exact-request identity must "
            "still come from User topic. Queries may use context terminology "
            "only when it remains consistent with User topic."
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
    normalized_feedback = " ".join(str(validation_feedback or "").split())[:500]
    if normalized_feedback:
        user += (
            "\nCORRECTIVE_RETRY: The prior JSON response failed strict local "
            "validation: "
            + json.dumps(normalized_feedback, ensure_ascii=True)
            + ". Correct that contract violation while preserving every original "
            "request constraint; return the complete JSON object again."
        )
    reservation: dict[str, object] = {}
    output_tokens = (
        PRACTICE_FAST_EXPAND_FALLBACK_OUTPUT_TOKENS
        if model == PRACTICE_FAST_EXPAND_FALLBACK_MODEL
        else PRACTICE_FAST_EXPAND_OUTPUT_TOKENS
    )
    try:
        if context is not None:
            reserved = context.reserve_gemini_call(
                operation="expansion",
                model=model,
                prompt_text=f"{_PRACTICE_FAST_SYSTEM}\n\n{user}",
                max_output_tokens=output_tokens,
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
                    response_json_schema=gemini_client._gemini3_json_schema(
                        _PracticeFastExpansion
                    ),
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    max_output_tokens=output_tokens,
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
                    status_code=gemini_client._gemini_status_code(exc),
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
    source_context: str | None = None,
    tried_queries: Sequence[str] = (),
    recovery_reason: str | None = None,
    rejected_video_ids: Sequence[str] = (),
    validation_feedback: str | None = None,
) -> str:
    return run_cancellable(
        lambda: _practice_fast_gemini_raw_async(
            topic,
            n,
            model=model,
            level=level,
            should_cancel=should_cancel,
            context=context,
            source_context=source_context,
            tried_queries=tried_queries,
            recovery_reason=recovery_reason,
            rejected_video_ids=rejected_video_ids,
            validation_feedback=validation_feedback,
        ),
        should_cancel,
    )


def _practice_fast_failure_is_retryable(exc: Exception) -> bool:
    """Retry transport/local-contract failures, provider 499, 408/429, and 5xx."""
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
        return status in {408, 429, 499} or 500 <= status <= 599

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


def _practice_fast_validation_feedback(exc: Exception) -> str:
    """Return compact structural feedback without replaying the rejected payload."""
    if not isinstance(exc, ValueError):
        return ""
    message = str(exc).split("[type=", 1)[0]
    return " ".join(message.split())[:500]


def _practice_fast_failure_log_summary(exc: Exception) -> str:
    """Describe a failure without logging rejected model or learner values."""
    parts = [f"type={type(exc).__name__}"]
    status = gemini_client._gemini_status_code(exc)
    if status is not None:
        parts.append(f"status={status}")
    validation_feedback = _practice_fast_validation_feedback(exc)
    if validation_feedback:
        parts.append(f"validation={validation_feedback}")
    return " ".join(parts)


def _practice_fast_provider_error(exc: Exception) -> ProviderError:
    """Preserve typed failures and classify raw Gemini expansion failures."""
    status = gemini_client._gemini_status_code(exc)
    retry_after_sec = gemini_client._gemini_retry_after(exc)
    if isinstance(exc, ProviderError) and (
        status is None or isinstance(exc, ProviderResponseValidationError)
    ):
        return exc
    detail = type(exc).__name__
    kwargs = {
        "provider": "gemini",
        "operation": "expansion",
        "status_code": status,
        "detail": detail,
    }
    if status in {401, 403}:
        return ProviderAuthenticationError(
            "Gemini authentication failed.", **kwargs
        )
    if status == 402:
        return ProviderQuotaError("Gemini quota is exhausted.", **kwargs)
    if status == 404:
        return ModelUnavailableError(
            "The configured Gemini expansion model is unavailable.", **kwargs
        )
    if status == 429:
        return ProviderRateLimitError(
            "Gemini expansion is rate limited.",
            retry_after_sec=retry_after_sec,
            **kwargs,
        )
    if status in {408, 499} or (status is not None and 500 <= status <= 599):
        return ProviderTransientError(
            "Gemini expansion is temporarily unavailable.", **kwargs
        )
    if isinstance(exc, ValueError):
        return ProviderResponseValidationError(
            "Gemini responded, but its query plan did not satisfy the "
            "learning-request contract.",
            **{**kwargs, "status_code": status or 200},
        )
    if status is None and _practice_fast_failure_is_retryable(exc):
        return ProviderTransientError("Could not reach Gemini expansion.", **kwargs)
    return ProviderRequestError("Gemini rejected the expansion request.", **kwargs)


def expand_query_practice_fast(
    topic: str,
    n: int,
    level: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    context: "GenerationContext | None" = None,
    source_context: str | None = None,
    tried_queries: Sequence[str] = (),
    recovery_reason: str | None = None,
    rejected_video_ids: Sequence[str] = (),
) -> dict:
    """Return cached Gemini search terms or raise a typed provider failure.

    Expansion deliberately does not reserve the Supadata search budget tracked by
    ``context``. The context enables durable cache reuse and usage accounting.
    """
    topic = " ".join(str(topic or "").split())
    count = max(0, int(n))
    if not topic or count == 0:
        return literal_fallback(topic, count)
    normalized_source_context = _normalized_source_context(source_context)
    normalized_tried = _normalize(list(tried_queries), len(tried_queries))
    tried_keys = {_key(query) for query in normalized_tried}
    recovering_zero_results = bool(normalized_tried)
    raise_if_cancelled(should_cancel)
    cache_key = (
        _expansion_cache_key(
            topic,
            count,
            level,
            source_context=normalized_source_context,
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
            cached = _without_tried_expansion_queries(cached, tried_keys)
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
    last_failure: Exception | None = None
    with singleflight(cache_key, should_cancel) if cache_key else nullcontext():
        cached = _read_cached_expansion(cache_key, count) if cache_key else None
        if cached is not None:
            if recovering_zero_results:
                cached = _without_tried_expansion_queries(cached, tried_keys)
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
        validation_feedback = ""
        use_fallback_model = False
        for attempt in range(PRACTICE_FAST_EXPAND_ATTEMPTS):
            raise_if_cancelled(should_cancel)
            model = (
                PRACTICE_FAST_EXPAND_FALLBACK_MODEL
                if use_fallback_model
                or attempt + 1 == PRACTICE_FAST_EXPAND_ATTEMPTS
                else PRACTICE_FAST_EXPAND_MODEL
            )
            try:
                kwargs = {
                    "model": model,
                    "level": level,
                    "should_cancel": should_cancel,
                }
                if context is not None:
                    kwargs["context"] = context
                if normalized_source_context:
                    kwargs["source_context"] = normalized_source_context
                if recovering_zero_results:
                    kwargs["tried_queries"] = normalized_tried
                    kwargs["recovery_reason"] = _normalized_recovery_reason(
                        recovery_reason
                    )
                    kwargs["rejected_video_ids"] = (
                        _normalized_rejected_video_ids(rejected_video_ids)
                    )
                if validation_feedback:
                    kwargs["validation_feedback"] = validation_feedback
                raw = _practice_fast_gemini_raw(topic, count, **kwargs)
                parsed = _PracticeFastExpansion.model_validate_json(raw)
                corrected = " ".join(str(parsed.corrected or topic).split()) or topic
                raw_query_metadata: list[dict[str, object]] = []
                queries = _validated_ai_queries(
                    topic,
                    parsed,
                    count,
                    query_metadata_out=raw_query_metadata,
                )
                if recovering_zero_results:
                    filtered = _without_tried_expansion_queries(
                        {
                            "queries": queries,
                            "query_metadata": raw_query_metadata,
                        },
                        tried_keys,
                    )
                    queries = filtered["queries"]
                    raw_query_metadata = filtered["query_metadata"]
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
                acquisition_ids = list(
                    parsed.acquisition_obligation_constraint_ids
                )
                query_metadata = _validated_query_metadata(
                    raw_query_metadata,
                    queries=queries,
                    intent_contract=intent_contract,
                    acquisition_constraint_ids=acquisition_ids,
                )
                if query_metadata is None:
                    raise ValueError(
                        "Gemini returned invalid query-to-intent metadata"
                    )
                result = {
                    "corrected": corrected,
                    "queries": queries,
                    "provider_used": "gemini",
                    "intent_contract": intent_contract,
                    "acquisition_obligation_constraint_ids": acquisition_ids,
                    "query_metadata": query_metadata,
                }
                if cache_key:
                    _write_cached_expansion(cache_key, result)
                return result
            except CancellationError:
                raise
            except gemini_client.GeminiCancelledError as exc:
                raise CancellationError("Generation cancelled.") from exc
            except Exception as exc:
                raise_if_cancelled(should_cancel)
                last_failure = exc
                validation_feedback = _practice_fast_validation_feedback(exc)
                provider_status = gemini_client._gemini_status_code(exc)
                errors.append(
                    f"{model} attempt {attempt + 1}: "
                    f"{_practice_fast_failure_log_summary(exc)}"
                )
                # A provider-wide quota window applies to every Gemini model.
                # Model fallback inside the same durable attempt only multiplies
                # identical 429s and consumes the bounded job-attempt ceiling
                # before delayed recovery can run.
                if provider_status == 429:
                    break
                if (
                    provider_status == 404
                    and model == PRACTICE_FAST_EXPAND_MODEL
                    and PRACTICE_FAST_EXPAND_FALLBACK_MODEL != model
                ):
                    use_fallback_model = True
                    continue
                if (
                    attempt + 1 >= PRACTICE_FAST_EXPAND_ATTEMPTS
                    or not _practice_fast_failure_is_retryable(exc)
                ):
                    break
    logger.warning(
        "practice-fast Gemini %s failed: %s",
        "zero-result recovery" if recovering_zero_results else "expansion",
        "; ".join(errors),
    )
    if last_failure is None:
        last_failure = RuntimeError("Gemini expansion ended without a response")
    raise _practice_fast_provider_error(last_failure) from last_failure


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
