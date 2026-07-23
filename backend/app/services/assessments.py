"""Adaptive recall-check persistence, readiness, and session lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ...concept_families import has_incompatible_gemini_concept_family_contract
from ..db import (
    dumps_json,
    execute_modify,
    fetch_all,
    fetch_one,
    now_iso,
    upsert,
)
from ..clip_engine.errors import CancellationError, ProviderError


# A rendered reel can report this tiny, non-completion fraction asynchronously so
# the running lesson organizer knows which streamed candidates may already be in
# the immutable viewport prefix. Quiz cadence still uses scrolls and completion.
ACTIVE_REEL_OPEN_FRACTION = 0.001
COMPLETION_FRACTION = 0.80
# Compatibility for reels with no version-2 organizer metadata at all.
LEGACY_CADENCE_TARGET = 3
SESSION_QUESTION_TARGET = 3
CONCEPT_ADJUSTMENT_BOUND = 0.25
BACKFILL_CACHE_PREFIX = "assessment_question_backfill:"
NEGATIVE_BACKFILL_CACHE_TTL_SECONDS = 30
RECALL_PREPARATION_TIMEOUT_SECONDS = 8.0

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_GROUNDING_STOP_WORDS = {
    "about", "after", "again", "also", "because", "before", "being", "between",
    "could", "does", "from", "have", "into", "more", "most", "other", "should",
    "than", "that", "their", "there", "these", "they", "this", "those", "through",
    "very", "what", "when", "where", "which", "while", "with", "would", "your",
}


class AssessmentCancelledError(Exception):
    pass


class _BackfillPlan(BaseModel):
    # Keep each item permissive so one malformed question can be discarded
    # without invalidating every other reel in the single batched call.
    questions: list[dict[str, Any]] = Field(default_factory=list)


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise AssessmentCancelledError("Assessment generation cancelled.")


def assessment_checkpoint_reel_ids(
    ordered_reel_ids: list[str],
    proposed_checkpoint_reel_ids: object = None,
    *,
    degraded: bool = False,
) -> list[str] | None:
    """Validate the organizer's exact checkpoint list without inventing cadence."""
    if (
        degraded
        or not ordered_reel_ids
        or not all(isinstance(reel_id, str) and reel_id for reel_id in ordered_reel_ids)
        or len(set(ordered_reel_ids)) != len(ordered_reel_ids)
        or not isinstance(proposed_checkpoint_reel_ids, list)
    ):
        return None
    positions = {reel_id: index for index, reel_id in enumerate(ordered_reel_ids)}
    proposed = proposed_checkpoint_reel_ids
    if (
        not all(isinstance(reel_id, str) and reel_id in positions for reel_id in proposed)
        or len(set(proposed)) != len(proposed)
        or [positions[reel_id] for reel_id in proposed]
        != sorted(positions[reel_id] for reel_id in proposed)
    ):
        return None
    return list(proposed)


@contextmanager
def _atomic_write(conn: Any):
    """Use a savepoint so multi-row assessment writes cannot be half-persisted."""
    owns_postgres_transaction = getattr(conn, "autocommit", None) is True
    if owns_postgres_transaction:
        conn.autocommit = False
    savepoint = f"assessment_write_{uuid.uuid4().hex}"
    try:
        execute_modify(conn, f"SAVEPOINT {savepoint}")
        try:
            yield
        except BaseException:
            execute_modify(conn, f"ROLLBACK TO SAVEPOINT {savepoint}")
            execute_modify(conn, f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            execute_modify(conn, f"RELEASE SAVEPOINT {savepoint}")
            if owns_postgres_transaction:
                conn.commit()
    finally:
        if owns_postgres_transaction:
            if not conn.autocommit:
                conn.rollback()
                conn.autocommit = True


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(value or "")
        if len(token) >= 3 and token.lower() not in _GROUNDING_STOP_WORDS
    }


def _question_fingerprint(row: dict[str, Any]) -> str:
    payload = "|".join(
        [
            str(row.get("reel_id") or row.get("id") or ""),
            f"{float(row.get('t_start') or 0.0):.3f}",
            f"{float(row.get('t_end') or 0.0):.3f}",
            str(row.get("transcript_snippet") or "").strip(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_question(
    *,
    prompt: object,
    options: object,
    correct_index: object,
    explanation: object,
    transcript: str,
) -> dict[str, Any] | None:
    clean_prompt = " ".join(str(prompt or "").split())[:600]
    clean_explanation = " ".join(str(explanation or "").split())[:1200]
    if not clean_prompt or not clean_explanation or not isinstance(options, list):
        return None
    clean_options = [" ".join(str(option or "").split())[:300] for option in options]
    if len(clean_options) != 4 or any(not option for option in clean_options):
        return None
    if len({option.casefold() for option in clean_options}) != 4:
        return None
    if isinstance(correct_index, bool):
        return None
    if isinstance(correct_index, float) and not correct_index.is_integer():
        return None
    try:
        answer_index = int(correct_index)
    except (TypeError, ValueError):
        return None
    if answer_index < 0 or answer_index >= 4:
        return None
    transcript_tokens = _meaningful_tokens(transcript)
    support_tokens = _meaningful_tokens(
        f"{clean_explanation} {clean_options[answer_index]}"
    )
    if not transcript_tokens or not transcript_tokens.intersection(support_tokens):
        return None
    return {
        "prompt": clean_prompt,
        "options": clean_options,
        "correct_index": answer_index,
        "explanation": clean_explanation,
    }


def _transcript_option(row: dict[str, Any], *, word_limit: int = 8) -> str:
    transcript = " ".join(str(row.get("transcript_snippet") or "").split())
    words = transcript.split()
    return " ".join(words[:word_limit]).strip(" ,.;:!?\"'")


def _grounded_fallback_question(
    row: dict[str, Any],
    *,
    alternative_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build an immediate transcript-only recall question without provider I/O."""
    transcript = " ".join(str(row.get("transcript_snippet") or "").split())
    words = transcript.split()
    if not words:
        return None
    supported_option = _transcript_option(row)
    support_quote = " ".join(words[:24]).strip()
    if not supported_option or not support_quote:
        return None
    distractors: list[str] = []
    supported_key = supported_option.casefold()
    seen = {supported_key}

    def add_distractor(option: str) -> None:
        clean_option = option.strip()
        option_key = clean_option.casefold()
        if (
            not clean_option
            or option_key in seen
            or supported_key.startswith(option_key)
            or option_key.startswith(supported_key)
        ):
            return
        distractors.append(clean_option)
        seen.add(option_key)

    for alternative in alternative_rows or []:
        if str(alternative.get("reel_id") or "") == str(row.get("reel_id") or ""):
            continue
        add_distractor(_transcript_option(alternative))
        if len(distractors) == 3:
            break
    for option in (
        "The clip contains only course scheduling details",
        "The clip is an advertisement with no explanation",
        "The clip makes the opposite claim without support",
        "The clip never discusses the lesson topic",
    ):
        if len(distractors) == 3:
            break
        add_distractor(option)
    if len(distractors) != 3:
        return None
    correct_index = int(_question_fingerprint(row)[:2], 16) % 4
    options = list(distractors)
    options.insert(correct_index, supported_option)
    return {
        "prompt": "Which exact excerpt begins this clip?",
        "options": options,
        "correct_index": correct_index,
        "explanation": support_quote,
    }


def store_reel_assessment_question(
    conn: Any,
    *,
    reel_id: str,
    prompt: str,
    options: list[str],
    correct_index: int,
    explanation: str,
    fingerprint: str | None = None,
) -> dict[str, Any] | None:
    """Validate and persist one private answer-bearing question for a reel."""
    reel = fetch_one(
        conn,
        """
        SELECT id AS reel_id, t_start, t_end, transcript_snippet
        FROM reels
        WHERE id = ?
        """,
        (reel_id,),
    )
    if not reel:
        return None
    stable_fingerprint = str(fingerprint or "").strip() or _question_fingerprint(reel)
    existing = fetch_one(
        conn,
        "SELECT * FROM reel_assessment_questions WHERE reel_id = ? AND fingerprint = ?",
        (reel_id, stable_fingerprint),
    )
    if existing:
        try:
            stored_options = json.loads(str(existing.get("options_json") or "[]"))
        except (TypeError, json.JSONDecodeError):
            stored_options = []
        return {
            "id": str(existing["id"]),
            "reel_id": reel_id,
            "fingerprint": stable_fingerprint,
            "prompt": str(existing.get("prompt") or ""),
            "options": stored_options if isinstance(stored_options, list) else [],
            "correct_index": int(existing.get("correct_index") or 0),
            "explanation": str(existing.get("explanation") or ""),
        }
    validated = _validated_question(
        prompt=prompt,
        options=options,
        correct_index=correct_index,
        explanation=explanation,
        transcript=str(reel.get("transcript_snippet") or ""),
    )
    if not validated:
        return None
    question_id = "assessment-question-" + hashlib.sha256(
        f"{reel_id}|{stable_fingerprint}".encode("utf-8")
    ).hexdigest()[:24]
    execute_modify(
        conn,
        """
        INSERT INTO reel_assessment_questions (
            id, reel_id, fingerprint, prompt, options_json,
            correct_index, explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(reel_id, fingerprint) DO NOTHING
        """,
        (
            question_id,
            reel_id,
            stable_fingerprint,
            validated["prompt"],
            dumps_json(validated["options"]),
            validated["correct_index"],
            validated["explanation"],
            now_iso(),
        ),
    )
    stored = fetch_one(
        conn,
        "SELECT * FROM reel_assessment_questions WHERE reel_id = ? AND fingerprint = ?",
        (reel_id, stable_fingerprint),
    )
    if not stored:
        return None
    try:
        stored_options = json.loads(str(stored.get("options_json") or "[]"))
    except (TypeError, json.JSONDecodeError):
        stored_options = []
    return {
        "id": str(stored["id"]),
        "reel_id": reel_id,
        "fingerprint": stable_fingerprint,
        "prompt": str(stored.get("prompt") or ""),
        "options": stored_options if isinstance(stored_options, list) else [],
        "correct_index": int(stored.get("correct_index") or 0),
        "explanation": str(stored.get("explanation") or ""),
    }


class AssessmentService:
    def __init__(
        self,
        question_generator: Callable[
            [list[dict[str, Any]], Callable[[], bool] | None], object
        ]
        | None = None,
    ) -> None:
        self._question_generator = question_generator

    def store_question(self, conn: Any, **kwargs: Any) -> dict[str, Any] | None:
        return store_reel_assessment_question(conn, **kwargs)

    @staticmethod
    def _information_units(rows: list[dict[str, Any]]) -> float:
        seen_concepts: set[str] = set()
        total = 0.0
        for row in rows:
            duration_minutes = max(
                0.35,
                min(1.75, max(0.0, float(row.get("t_end") or 0.0) - float(row.get("t_start") or 0.0)) / 60.0),
            )
            try:
                informativeness = float(row.get("informativeness"))
            except (TypeError, ValueError):
                informativeness = 0.6
            informativeness = max(0.6, min(1.0, informativeness))
            concept_id = str(row.get("concept_id") or "")
            novelty = 0.85 if concept_id in seen_concepts else 1.15
            if concept_id:
                seen_concepts.add(concept_id)
            total += duration_minutes * informativeness * novelty
        return total

    @staticmethod
    def _organizer_checkpoint_assignments(
        conn: Any,
        *,
        learner_id: str,
        material_id: str,
        reel_ids: set[str],
    ) -> dict[str, tuple[bool, bool]]:
        """Resolve reels against the newest learner-owned released lesson plans.

        Each value is ``(organizer_controls_cadence, is_checkpoint)``. A valid
        version-2 plan controls only reels named in its order. Corrupt or
        incomplete version-2 metadata controls the unresolved current window
        conservatively, so it cannot accidentally fall back to a numeric quiz.
        """
        if not reel_ids:
            return {}
        rows = fetch_all(
            conn,
            """
            SELECT g.lesson_order_json
            FROM reel_generation_jobs j
            JOIN reel_generations g ON g.id = j.result_generation_id
            WHERE j.learner_id = ?
              AND j.material_id = ?
              AND j.status IN ('completed', 'partial')
              AND j.result_generation_id IS NOT NULL
            ORDER BY COALESCE(j.completed_at, j.updated_at, j.created_at) DESC,
                     j.created_at DESC,
                     j.id DESC
            LIMIT 100
            """,
            (learner_id, material_id),
        )
        unresolved = set(reel_ids)
        assignments: dict[str, tuple[bool, bool]] = {}
        for row in rows:
            raw = row.get("lesson_order_json")
            if raw is None:
                continue
            try:
                payload = json.loads(str(raw))
            except (TypeError, json.JSONDecodeError):
                for reel_id in unresolved:
                    assignments[reel_id] = (True, False)
                break
            if not isinstance(payload, dict):
                for reel_id in unresolved:
                    assignments[reel_id] = (True, False)
                break
            if payload.get("version") != 2:
                continue
            ordered = payload.get("ordered_reel_ids")
            if (
                not isinstance(ordered, list)
                or not ordered
                or not all(isinstance(reel_id, str) and reel_id for reel_id in ordered)
                or len(set(ordered)) != len(ordered)
            ):
                for reel_id in unresolved:
                    assignments[reel_id] = (True, False)
                break
            matching = unresolved.intersection(ordered)
            if not matching:
                continue
            if (
                payload.get("degraded") is True
                and "assessment_checkpoint_reel_ids" in payload
                and payload.get("assessment_checkpoint_reel_ids") is None
            ):
                # A degraded plan still identifies its own reels, but its null
                # checkpoint contract is not authoritative. Keep older plans
                # from reclaiming these reels and use the legacy cadence.
                for reel_id in matching:
                    assignments[reel_id] = (False, False)
                unresolved.difference_update(matching)
                if not unresolved:
                    break
                continue
            checkpoint_set = set(
                assessment_checkpoint_reel_ids(
                    ordered,
                    payload.get("assessment_checkpoint_reel_ids"),
                    degraded=payload.get("degraded") is not False,
                )
                or []
            )
            for reel_id in matching:
                assignments[reel_id] = (True, reel_id in checkpoint_set)
            unresolved.difference_update(matching)
            if not unresolved:
                break
        return assignments

    @classmethod
    def _cadence_target(
        cls,
        conn: Any,
        *,
        learner_id: str,
        material_id: str,
        window_cutoff: str,
        recent_accuracy: float | None,
        scroll_rows: list[dict[str, Any]],
    ) -> tuple[int, str | None, bool]:
        """Return reached position, checkpoint ID, and organizer cadence control."""
        del window_cutoff, recent_accuracy
        reel_ids = {
            str(row.get("reel_id") or "")
            for row in scroll_rows
            if str(row.get("reel_id") or "")
        }
        assignments = cls._organizer_checkpoint_assignments(
            conn,
            learner_id=learner_id,
            material_id=material_id,
            reel_ids=reel_ids,
        )
        organizer_controls_window = any(
            assignments.get(str(row.get("reel_id") or ""), (False, False))[0]
            for row in scroll_rows
        )
        if organizer_controls_window:
            degraded_positions: list[int] = []
            for position, row in enumerate(scroll_rows, start=1):
                reel_id = str(row.get("reel_id") or "")
                if assignments.get(reel_id, (False, False))[1]:
                    return position, reel_id, True
                if reel_id in assignments and not assignments[reel_id][0]:
                    degraded_positions.append(position)
            if len(degraded_positions) >= LEGACY_CADENCE_TARGET:
                return degraded_positions[LEGACY_CADENCE_TARGET - 1], None, False
            return 0, None, True
        return LEGACY_CADENCE_TARGET, None, False

    @staticmethod
    def _cadence_session_id(
        *, learner_id: str, material_id: str, cadence_cutoff: str
    ) -> str:
        window_key = json.dumps(
            [learner_id, material_id, cadence_cutoff],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"studyreels-assessment-session:{window_key}",
            )
        )

    def _accuracy_stats(
        self, conn: Any, learner_id: str, material_id: str
    ) -> tuple[float | None, float | None, list[float]]:
        rows = fetch_all(
            conn,
            """
            SELECT correct_count, question_count
            FROM assessment_sessions
            WHERE learner_id = ? AND material_id = ? AND status = 'completed'
            ORDER BY completed_at DESC, created_at DESC
            LIMIT 3
            """,
            (learner_id, material_id),
        )
        if not rows:
            return None, None, []
        accuracies = [
            float(row.get("correct_count") or 0) / max(1, int(row.get("question_count") or 0))
            for row in rows
        ]
        total_correct = sum(int(row.get("correct_count") or 0) for row in rows)
        total_questions = sum(int(row.get("question_count") or 0) for row in rows)
        rolling = total_correct / max(1, total_questions)
        return accuracies[0], rolling, accuracies

    @staticmethod
    def _latest_cutoff(conn: Any, learner_id: str, material_id: str, status: str) -> str:
        column = "completed_at" if status == "completed" else "snoozed_at"
        row = fetch_one(
            conn,
            f"""
            SELECT {column} AS cutoff
            FROM assessment_sessions
            WHERE learner_id = ? AND material_id = ? AND status = ? AND {column} IS NOT NULL
            ORDER BY {column} DESC
            LIMIT 1
            """,
            (learner_id, material_id, status),
        )
        return str((row or {}).get("cutoff") or "")

    @staticmethod
    def _latest_cadence_cutoff(
        conn: Any,
        learner_id: str,
        material_id: str,
        status: str,
    ) -> str:
        """Consume through the organizer checkpoint, not later session completion."""
        column = "completed_at" if status == "completed" else "snoozed_at"
        row = fetch_one(
            conn,
            f"""
            SELECT CASE
                     WHEN TRIM(COALESCE(s.organizer_checkpoint_reel_id, '')) != ''
                     THEN COALESCE(p.scrolled_at, s.{column}, s.created_at)
                     ELSE s.{column}
                   END AS cutoff
            FROM assessment_sessions s
            LEFT JOIN learner_reel_progress p
              ON p.learner_id = s.learner_id
             AND p.reel_id = s.organizer_checkpoint_reel_id
            WHERE s.learner_id = ?
              AND s.material_id = ?
              AND s.status = ?
              AND s.{column} IS NOT NULL
            ORDER BY cutoff DESC
            LIMIT 1
            """,
            (learner_id, material_id, status),
        )
        return str((row or {}).get("cutoff") or "")

    @staticmethod
    def _completed_rows(
        conn: Any, learner_id: str, material_id: str, after: str = ""
    ) -> list[dict[str, Any]]:
        return fetch_all(
            conn,
            """
            SELECT
                p.completed_at,
                r.id AS reel_id,
                r.concept_id,
                r.video_id,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                r.informativeness,
                r.difficulty,
                c.title AS concept_title
            FROM learner_reel_progress p
            JOIN reels r ON r.id = p.reel_id
            JOIN concepts c ON c.id = r.concept_id
            WHERE p.learner_id = ?
              AND p.material_id = ?
              AND p.completed_at IS NOT NULL
              AND (? = '' OR p.completed_at > ?)
            ORDER BY p.completed_at ASC, r.id ASC
            """,
            (learner_id, material_id, after, after),
        )

    @staticmethod
    def _scrolled_rows(
        conn: Any, learner_id: str, material_id: str, after: str = ""
    ) -> list[dict[str, Any]]:
        return fetch_all(
            conn,
            """
            SELECT
                p.scrolled_at,
                r.id AS reel_id,
                r.concept_id,
                r.video_id,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                r.informativeness,
                r.difficulty,
                c.title AS concept_title
            FROM learner_reel_progress p
            JOIN reels r ON r.id = p.reel_id
            JOIN concepts c ON c.id = r.concept_id
            WHERE p.learner_id = ?
              AND p.material_id = ?
              AND p.scrolled_at IS NOT NULL
              AND (? = '' OR p.scrolled_at > ?)
            ORDER BY p.scrolled_at ASC, r.id ASC
            """,
            (learner_id, material_id, after, after),
        )

    @staticmethod
    def _pending_row(conn: Any, learner_id: str, material_id: str) -> dict[str, Any] | None:
        return fetch_one(
            conn,
            """
            SELECT *
            FROM assessment_sessions
            WHERE learner_id = ? AND material_id = ? AND status = 'pending'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (learner_id, material_id),
        )

    @staticmethod
    def _available_questions(
        conn: Any, learner_id: str, material_id: str, after: str
    ) -> list[dict[str, Any]]:
        rows = fetch_all(
            conn,
            """
            SELECT
                q.id,
                q.reel_id,
                q.fingerprint,
                q.prompt,
                q.options_json,
                q.created_at,
                r.concept_id,
                r.video_id,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                r.difficulty,
                c.title AS concept_title,
                p.scrolled_at
            FROM reel_assessment_questions q
            JOIN reels r ON r.id = q.reel_id
            JOIN concepts c ON c.id = r.concept_id
            JOIN learner_reel_progress p
              ON p.reel_id = r.id
             AND p.learner_id = ?
            WHERE r.material_id = ?
              AND p.scrolled_at IS NOT NULL
              AND (? = '' OR p.scrolled_at > ?)
              AND NOT EXISTS (
                  SELECT 1
                  FROM assessment_attempts a
                  WHERE a.learner_id = ? AND a.question_id = q.id
              )
            ORDER BY p.scrolled_at DESC, q.created_at DESC, q.id ASC
            """,
            (learner_id, material_id, after, after, learner_id),
        )
        for row in rows:
            try:
                options = json.loads(str(row.get("options_json") or "[]"))
            except (TypeError, json.JSONDecodeError):
                options = []
            row["options"] = options if isinstance(options, list) else []
        return [
            row
            for row in rows
            if len(row.get("options") or []) == 4
            and str(row.get("fingerprint") or "") == _question_fingerprint(row)
        ]

    def _readiness_state(self, conn: Any, learner_id: str, material_id: str) -> dict[str, Any]:
        pending = self._pending_row(conn, learner_id, material_id)
        recent_accuracy, rolling_accuracy, recent_sessions = self._accuracy_stats(
            conn, learner_id, material_id
        )
        completed_cutoff = self._latest_cutoff(conn, learner_id, material_id, "completed")
        completed_cadence_cutoff = self._latest_cadence_cutoff(
            conn, learner_id, material_id, "completed"
        )
        snoozed_cadence_cutoff = self._latest_cadence_cutoff(
            conn, learner_id, material_id, "snoozed"
        )
        cadence_cutoff = max(completed_cadence_cutoff, snoozed_cadence_cutoff)
        completed_rows = self._completed_rows(conn, learner_id, material_id, completed_cutoff)
        total_units = self._information_units(completed_rows)
        scroll_rows = self._scrolled_rows(conn, learner_id, material_id, cadence_cutoff)
        cadence_target, organizer_checkpoint_reel_id, organizer_plan_active = self._cadence_target(
            conn,
            learner_id=learner_id,
            material_id=material_id,
            window_cutoff=cadence_cutoff,
            recent_accuracy=recent_accuracy,
            scroll_rows=scroll_rows,
        )

        available = self._available_questions(conn, learner_id, material_id, cadence_cutoff)
        if organizer_checkpoint_reel_id:
            available = [
                row
                for row in available
                if str(row.get("reel_id") or "")
                == organizer_checkpoint_reel_id
            ]
        session_question_target = (
            self._pending_session_target(pending)
            if pending
            else 1
            if organizer_checkpoint_reel_id
            else SESSION_QUESTION_TARGET
        )
        question_reels = {
            str(row.get("reel_id") or "")
            for row in available
            if str(row.get("reel_id") or "")
        }
        question_pool_complete = len(question_reels) >= session_question_target
        numeric_due = bool(organizer_checkpoint_reel_id) or (
            not organizer_plan_active and len(scroll_rows) >= cadence_target
        )
        # Readiness is the durable due state. Question availability is handled
        # separately so a transient provider failure cannot erase that state.
        ready = bool(pending) or numeric_due
        return {
            "assessment_ready": ready,
            "numeric_due": numeric_due,
            "information_units": total_units,
            "readiness_threshold": float(cadence_target),
            "cadence_target": cadence_target,
            "session_question_target": session_question_target,
            "organizer_checkpoint_reel_id": organizer_checkpoint_reel_id,
            "organizer_plan_active": organizer_plan_active,
            "scroll_count": len(scroll_rows),
            "completed_cutoff": completed_cutoff,
            "cadence_cutoff": cadence_cutoff,
            "completion_rows": completed_rows,
            "scroll_rows": scroll_rows,
            "available_questions": available,
            "question_pool_complete": question_pool_complete,
            "pending": pending,
            "recent_accuracy": recent_accuracy,
            "rolling_accuracy": rolling_accuracy,
            "recent_session_accuracies": recent_sessions,
        }

    def record_progress(
        self,
        conn: Any,
        *,
        learner_id: str,
        reel_id: str,
        max_fraction: float,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        _check_cancelled(should_cancel)
        reel = fetch_one(conn, "SELECT id, material_id FROM reels WHERE id = ?", (reel_id,))
        if not reel:
            raise ValueError("reel_id not found")
        existing = fetch_one(
            conn,
            "SELECT * FROM learner_reel_progress WHERE learner_id = ? AND reel_id = ?",
            (learner_id, reel_id),
        )
        fraction = max(0.0, min(1.0, float(max_fraction)))
        prior_fraction = float((existing or {}).get("max_fraction") or 0.0)
        stored_fraction = max(prior_fraction, fraction)
        prior_completed_at = str((existing or {}).get("completed_at") or "")
        newly_completed = not prior_completed_at and stored_fraction >= COMPLETION_FRACTION
        timestamp = now_iso()
        completed_at = prior_completed_at or (timestamp if newly_completed else None)
        _check_cancelled(should_cancel)
        upsert(
            conn,
            "learner_reel_progress",
            {
                "learner_id": learner_id,
                "reel_id": reel_id,
                "material_id": str(reel["material_id"]),
                "max_fraction": stored_fraction,
                "completed_at": completed_at,
                "created_at": str((existing or {}).get("created_at") or timestamp),
                "updated_at": timestamp,
            },
            pk=["learner_id", "reel_id"],
        )
        state = self._readiness_state(conn, learner_id, str(reel["material_id"]))
        return {
            "reel_id": reel_id,
            "completed": bool(completed_at),
            "newly_completed": newly_completed,
            "assessment_ready": bool(state["assessment_ready"]),
            "information_units": round(float(state["information_units"]), 4),
            "readiness_threshold": round(float(state["readiness_threshold"]), 4),
        }

    def record_scroll(
        self,
        conn: Any,
        *,
        learner_id: str,
        reel_id: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Record one distinct forward navigation without changing watch analytics."""
        _check_cancelled(should_cancel)
        reel = fetch_one(conn, "SELECT id, material_id FROM reels WHERE id = ?", (reel_id,))
        if not reel:
            raise ValueError("reel_id not found")
        material_id = str(reel["material_id"])
        timestamp = now_iso()
        inserted = execute_modify(
            conn,
            """
            INSERT INTO learner_reel_progress (
                learner_id, reel_id, material_id, max_fraction,
                scrolled_at, completed_at, created_at, updated_at
            ) VALUES (?, ?, ?, 0.0, ?, NULL, ?, ?)
            ON CONFLICT(learner_id, reel_id) DO NOTHING
            """,
            (learner_id, reel_id, material_id, timestamp, timestamp, timestamp),
        )
        updated = 0
        if inserted <= 0:
            updated = execute_modify(
                conn,
                """
                UPDATE learner_reel_progress
                SET scrolled_at = ?, updated_at = ?
                WHERE learner_id = ? AND reel_id = ? AND scrolled_at IS NULL
                """,
                (timestamp, timestamp, learner_id, reel_id),
            )
        newly_scrolled = inserted > 0 or updated > 0
        _check_cancelled(should_cancel)
        state = self._readiness_state(conn, learner_id, material_id)
        return {
            "reel_id": reel_id,
            "material_id": material_id,
            "newly_scrolled": newly_scrolled,
            "assessment_ready": bool(state["assessment_ready"]),
            "scroll_count": int(state["scroll_count"]),
            "cadence_target": int(state["cadence_target"]),
        }

    def _load_cached_backfill(self, conn: Any, fingerprint: str) -> tuple[bool, object]:
        row = fetch_one(
            conn,
            "SELECT response_json, created_at FROM llm_cache WHERE cache_key = ?",
            (f"{BACKFILL_CACHE_PREFIX}{fingerprint}",),
        )
        if not row:
            return False, None
        try:
            payload = json.loads(str(row.get("response_json") or "{}"))
        except json.JSONDecodeError:
            return False, None
        if not isinstance(payload, dict) or "question" not in payload:
            return False, None
        question = payload.get("question")
        if question is None:
            try:
                created_at = datetime.fromisoformat(
                    str(row.get("created_at") or "").replace("Z", "+00:00")
                )
            except ValueError:
                return False, None
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age_seconds >= NEGATIVE_BACKFILL_CACHE_TTL_SECONDS:
                return False, None
        return True, question

    @staticmethod
    def _write_cached_backfill(conn: Any, fingerprint: str, question: object) -> None:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": f"{BACKFILL_CACHE_PREFIX}{fingerprint}",
                "response_json": dumps_json({"question": question}),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )

    def _default_question_generator(
        self,
        rows: list[dict[str, Any]],
        should_cancel: Callable[[], bool] | None,
    ) -> object:
        from ..clip_engine.clipper import config as clipper_config
        from ..clip_engine.clipper.llm import llm_json

        clips = [
            {
                "reel_id": row["reel_id"],
                "concept": row.get("concept_title") or "",
                "transcript": str(row.get("transcript_snippet") or "")[:1800],
            }
            for row in rows
        ]
        plan = llm_json(
            (
                "Use only the supplied transcript text; no video, image, audio, frame, or "
                "other media is attached or permitted. Create one grounded four-option "
                "recall question for each supplied clip. "
                "Every option must be distinct. The explanation must be supported by that clip's transcript. "
                "Keep each prompt at most 16 words, each option at most 8 words, and each "
                "explanation one sentence and at most 24 words. "
                "Use LaTeX only when it improves mathematical clarity; wrap inline math in "
                "\\( ... \\) and display math in \\[ ... \\]. Because the response is JSON, "
                "escape every LaTeX backslash in string values (for example, emit `\\\\(` "
                "for an inline opening delimiter). "
                "Return JSON only as {\"questions\":[{\"reel_id\":...,\"prompt\":...,"
                "\"options\":[... exactly four ...],\"correct_index\":0-3,\"explanation\":...}]}"
            ),
            dumps_json({"clips": clips}),
            _BackfillPlan,
            temperature=0.2,
            model=clipper_config.ASSESSMENT_MODEL,
            max_output_tokens=max(2048, min(8192, len(rows) * 700)),
            should_cancel=should_cancel,
        )
        return plan.model_dump()

    def _ensure_question_pool(
        self,
        conn: Any,
        *,
        learner_id: str,
        material_id: str,
        source_rows: list[dict[str, Any]],
        should_cancel: Callable[[], bool] | None,
        question_target: int | None = None,
        write_negative_cache: bool = True,
    ) -> None:
        self._ensure_question_pool_core(
            conn,
            learner_id=learner_id,
            material_id=material_id,
            source_rows=source_rows,
            should_cancel=should_cancel,
            question_target=question_target,
            write_negative_cache=write_negative_cache,
        )

    def _ensure_question_pool_core(
        self,
        conn: Any,
        *,
        learner_id: str,
        material_id: str,
        source_rows: list[dict[str, Any]],
        should_cancel: Callable[[], bool] | None,
        question_target: int | None,
        write_negative_cache: bool,
    ) -> None:
        del learner_id, material_id  # content is reel-scoped; eligibility is learner-scoped.
        uncached: list[dict[str, Any]] = []
        satisfied_reels: set[str] = set()
        satisfied_concepts: set[str] = set()
        satisfied_videos: set[str] = set()
        newest_first = list(reversed(source_rows))
        distinct_concepts: list[dict[str, Any]] = []
        repeated_concepts: list[dict[str, Any]] = []
        seen_concepts: set[str] = set()
        for row in newest_first:
            concept_id = str(row.get("concept_id") or "")
            if concept_id and concept_id not in seen_concepts:
                distinct_concepts.append(row)
                seen_concepts.add(concept_id)
            else:
                repeated_concepts.append(row)
        requested_target = (
            SESSION_QUESTION_TARGET
            if question_target is None
            else max(0, int(question_target))
        )
        effective_target = min(
            requested_target,
            len(
                {
                    str(row.get("reel_id") or row.get("id") or "")
                    for row in source_rows
                    if str(row.get("reel_id") or row.get("id") or "")
                }
            ),
        )
        if effective_target <= 0:
            return

        def pool_complete() -> bool:
            return effective_target > 0 and len(satisfied_reels) >= effective_target

        candidates: list[dict[str, Any]] = []
        for row in [*distinct_concepts, *repeated_concepts]:
            _check_cancelled(should_cancel)
            fingerprint = _question_fingerprint(row)
            existing = fetch_one(
                conn,
                "SELECT id FROM reel_assessment_questions WHERE reel_id = ? AND fingerprint = ?",
                (row["reel_id"], fingerprint),
            )
            if existing:
                satisfied_reels.add(str(row["reel_id"]))
                satisfied_concepts.add(str(row.get("concept_id") or ""))
                satisfied_videos.add(str(row.get("video_id") or ""))
                continue
            candidates.append({**row, "fingerprint": fingerprint})
        if pool_complete():
            return

        candidates.sort(
            key=lambda row: (
                int(str(row.get("concept_id") or "") not in satisfied_concepts)
                + int(str(row.get("video_id") or "") not in satisfied_videos),
                int(str(row.get("concept_id") or "") not in satisfied_concepts),
                int(str(row.get("video_id") or "") not in satisfied_videos),
                str(row.get("scrolled_at") or ""),
            ),
            reverse=True,
        )

        for row in candidates:
            _check_cancelled(should_cancel)
            fingerprint = str(row["fingerprint"])
            cache_hit, cached_question = self._load_cached_backfill(conn, fingerprint)
            if cache_hit:
                if isinstance(cached_question, dict):
                    _check_cancelled(should_cancel)
                    stored = store_reel_assessment_question(
                        conn,
                        reel_id=str(row["reel_id"]),
                        prompt=str(cached_question.get("prompt") or ""),
                        options=list(cached_question.get("options") or []),
                        correct_index=cached_question.get("correct_index"),
                        explanation=str(cached_question.get("explanation") or ""),
                        fingerprint=fingerprint,
                    )
                    if stored:
                        satisfied_reels.add(str(row["reel_id"]))
                        satisfied_concepts.add(str(row.get("concept_id") or ""))
                        satisfied_videos.add(str(row.get("video_id") or ""))
                        if pool_complete():
                            return
                continue
            uncached.append(row)
            projected_reels = satisfied_reels | {
                str(candidate.get("reel_id") or "") for candidate in uncached
            }
            if len(projected_reels) >= effective_target:
                break
        if not uncached:
            return
        _check_cancelled(should_cancel)
        generator = self._question_generator or self._default_question_generator
        try:
            try:
                generated = generator(uncached, should_cancel)
            except TypeError as two_arg_error:
                # Backward-compatible injection hook for older focused tests.
                try:
                    generated = generator(uncached)  # type: ignore[call-arg]
                except TypeError:
                    raise two_arg_error
        except CancellationError as exc:
            raise AssessmentCancelledError(str(exc)) from exc
        except ProviderError:
            # The due state is already durable. Leave it intact and retry generation
            # on the next session call.
            return
        _check_cancelled(should_cancel)
        if generated is None:
            return
        if isinstance(generated, dict):
            raw_questions = generated.get("questions")
        else:
            raw_questions = generated
        if not isinstance(raw_questions, list):
            return
        by_reel = {
            str(item.get("reel_id") or ""): item
            for item in raw_questions
            if isinstance(item, dict) and str(item.get("reel_id") or "")
        }
        for row in uncached:
            _check_cancelled(should_cancel)
            item = by_reel.get(str(row["reel_id"]))
            cached_value: dict[str, Any] | None = None
            if isinstance(item, dict):
                validated = _validated_question(
                    prompt=item.get("prompt"),
                    options=item.get("options"),
                    correct_index=item.get("correct_index"),
                    explanation=item.get("explanation"),
                    transcript=str(row.get("transcript_snippet") or ""),
                )
                if validated:
                    cached_value = validated
                    _check_cancelled(should_cancel)
                    store_reel_assessment_question(
                        conn,
                        reel_id=str(row["reel_id"]),
                        fingerprint=str(row["fingerprint"]),
                        **validated,
                    )
            _check_cancelled(should_cancel)
            if cached_value is not None or write_negative_cache:
                self._write_cached_backfill(
                    conn, str(row["fingerprint"]), cached_value
                )

    def prepare_reel_questions(
        self,
        conn: Any,
        *,
        reel_ids: list[str],
        should_cancel: Callable[[], bool] | None = None,
        use_model: bool = True,
    ) -> dict[str, int]:
        """Prepare one private recall question for every reel in a released batch."""
        normalized_ids = list(
            dict.fromkeys(
                clean_id
                for value in reel_ids
                if (clean_id := str(value or "").strip())
            )
        )
        if not normalized_ids:
            return {"requested": 0, "prepared": 0, "fallback": 0}
        placeholders = ", ".join(["?"] * len(normalized_ids))
        rows = fetch_all(
            conn,
            f"""
            SELECT
                r.id AS reel_id,
                r.concept_id,
                r.video_id,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                c.title AS concept_title
            FROM reels r
            JOIN concepts c ON c.id = r.concept_id
            WHERE r.id IN ({placeholders})
            """,
            tuple(normalized_ids),
        )
        by_id = {str(row.get("reel_id") or ""): row for row in rows}
        ordered_rows = [by_id[reel_id] for reel_id in normalized_ids if reel_id in by_id]
        fallback_count = 0
        if use_model:
            preparation_deadline = time.monotonic() + RECALL_PREPARATION_TIMEOUT_SECONDS

            def preparation_cancelled() -> bool:
                return bool(
                    (should_cancel is not None and should_cancel())
                    or time.monotonic() >= preparation_deadline
                )

            try:
                self._ensure_question_pool_core(
                    conn,
                    learner_id="",
                    material_id="",
                    source_rows=ordered_rows,
                    should_cancel=preparation_cancelled,
                    question_target=len(ordered_rows),
                    write_negative_cache=False,
                )
            except AssessmentCancelledError:
                if should_cancel is not None and should_cancel():
                    raise
            except Exception:
                # Provider-backed enrichment is optional. The deterministic
                # transcript questions below are the readiness guarantee.
                pass
        for row in ordered_rows:
            _check_cancelled(should_cancel)
            fingerprint = _question_fingerprint(row)
            existing = fetch_one(
                conn,
                "SELECT id FROM reel_assessment_questions "
                "WHERE reel_id = ? AND fingerprint = ?",
                (row["reel_id"], fingerprint),
            )
            if existing:
                continue
            fallback = _grounded_fallback_question(
                row,
                alternative_rows=ordered_rows,
            )
            if fallback and store_reel_assessment_question(
                conn,
                reel_id=str(row["reel_id"]),
                fingerprint=fingerprint,
                **fallback,
            ):
                fallback_count += 1
        prepared = sum(
            1
            for row in ordered_rows
            if fetch_one(
                conn,
                "SELECT id FROM reel_assessment_questions "
                "WHERE reel_id = ? AND fingerprint = ?",
                (row["reel_id"], _question_fingerprint(row)),
            )
        )
        return {
            "requested": len(ordered_rows),
            "prepared": prepared,
            "fallback": fallback_count,
        }

    def _desired_question_count(self, conn: Any, state: dict[str, Any]) -> int:
        del conn
        distinct_reels = {
            str(row.get("reel_id") or "")
            for row in state["available_questions"]
            if str(row.get("reel_id") or "")
        }
        return min(
            max(1, int(state.get("session_question_target") or SESSION_QUESTION_TARGET)),
            len(distinct_reels),
        )

    @staticmethod
    def _pending_session_target(pending: dict[str, Any]) -> int:
        """Each organizer checkpoint is one immediate reel-grounded question."""
        if str(pending.get("organizer_checkpoint_reel_id") or ""):
            return 1
        return SESSION_QUESTION_TARGET

    @staticmethod
    def _select_questions(
        rows: list[dict[str, Any]],
        count: int,
        *,
        existing_concepts: set[str] | None = None,
        existing_reels: set[str] | None = None,
        existing_videos: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        remaining = list(rows)
        chosen: list[dict[str, Any]] = []
        concepts = set(existing_concepts or ())
        reels = set(existing_reels or ())
        videos = set(existing_videos or ())
        while remaining and len(chosen) < count:
            distinct_rows = [
                row
                for row in remaining
                if str(row.get("reel_id") or "") not in reels
            ]
            if not distinct_rows:
                break
            best = max(
                distinct_rows,
                key=lambda row: (
                    int(str(row.get("concept_id") or "") not in concepts)
                    + int(str(row.get("video_id") or "") not in videos),
                    int(str(row.get("concept_id") or "") not in concepts),
                    int(str(row.get("video_id") or "") not in videos),
                    str(row.get("scrolled_at") or ""),
                    str(row.get("id") or ""),
                ),
            )
            remaining.remove(best)
            chosen.append(best)
            concepts.add(str(best.get("concept_id") or ""))
            reels.add(str(best.get("reel_id") or ""))
            videos.add(str(best.get("video_id") or ""))
        return chosen

    @staticmethod
    def _pending_question_rows(conn: Any, session_id: str) -> list[dict[str, Any]]:
        return fetch_all(
            conn,
            """
            SELECT
                sq.position,
                q.id,
                q.reel_id,
                r.concept_id,
                r.video_id
            FROM assessment_session_questions sq
            JOIN reel_assessment_questions q ON q.id = sq.question_id
            JOIN reels r ON r.id = q.reel_id
            WHERE sq.session_id = ?
            ORDER BY sq.position ASC
            """,
            (session_id,),
        )

    def _repair_pending_session(
        self,
        conn: Any,
        *,
        state: dict[str, Any],
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        pending = state.get("pending")
        if not pending:
            return None
        session_id = str(pending["id"])
        existing = self._pending_question_rows(conn, session_id)
        question_target = self._pending_session_target(pending)
        if len(existing) >= question_target:
            if int(pending.get("question_count") or 0) != len(existing):
                execute_modify(
                    conn,
                    "UPDATE assessment_sessions SET question_count = ?, updated_at = ? WHERE id = ?",
                    (len(existing), now_iso(), session_id),
                )
            return fetch_one(
                conn, "SELECT * FROM assessment_sessions WHERE id = ?", (session_id,)
            )

        needed = question_target - len(existing)
        additions = self._select_questions(
            state["available_questions"],
            needed,
            existing_concepts={str(row.get("concept_id") or "") for row in existing},
            existing_reels={str(row.get("reel_id") or "") for row in existing},
            existing_videos={str(row.get("video_id") or "") for row in existing},
        )
        if len(additions) != needed:
            return pending

        _check_cancelled(should_cancel)
        with _atomic_write(conn):
            for offset, question in enumerate(additions, start=len(existing)):
                _check_cancelled(should_cancel)
                execute_modify(
                    conn,
                    """
                    INSERT INTO assessment_session_questions (session_id, question_id, position)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, question["id"], offset),
                )
            _check_cancelled(should_cancel)
            execute_modify(
                conn,
                """
                UPDATE assessment_sessions
                SET question_count = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (question_target, now_iso(), session_id),
            )
        return fetch_one(
            conn, "SELECT * FROM assessment_sessions WHERE id = ?", (session_id,)
        )

    def _exposable_pending_session(
        self, conn: Any, pending: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        session = self._serialize_session(conn, pending)
        question_target = self._pending_session_target(pending or {})
        if (
            session
            and len(session["questions"]) >= question_target
            and int(session["question_count"]) == len(session["questions"])
        ):
            return session
        return None

    def pending(self, conn: Any, *, learner_id: str, material_id: str) -> dict[str, Any]:
        state = self._readiness_state(conn, learner_id, material_id)
        pending = self._repair_pending_session(conn, state=state)
        session = self._exposable_pending_session(conn, pending)
        return {
            "status": "pending" if session else "none",
            "assessment_ready": bool(session or state["assessment_ready"]),
            "session": session,
            "recent_accuracy": state["recent_accuracy"],
            "rolling_accuracy": state["rolling_accuracy"],
        }

    def next_session(
        self,
        conn: Any,
        *,
        learner_id: str,
        material_id: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        state = self._readiness_state(conn, learner_id, material_id)
        if state["pending"]:
            if not self._exposable_pending_session(conn, state["pending"]):
                checkpoint_reel_id = str(
                    state["pending"].get("organizer_checkpoint_reel_id") or ""
                )
                self.prepare_reel_questions(
                    conn,
                    reel_ids=[checkpoint_reel_id] if checkpoint_reel_id else [
                        str(row.get("reel_id") or "")
                        for row in state["scroll_rows"]
                    ],
                    should_cancel=should_cancel,
                    use_model=False,
                )
                state = self._readiness_state(conn, learner_id, material_id)
                state["pending"] = self._repair_pending_session(
                    conn, state=state, should_cancel=should_cancel
                )
            session = self._exposable_pending_session(conn, state["pending"])
            if not session:
                return {
                    "status": "not_ready",
                    "assessment_ready": True,
                    "session": None,
                    "recent_accuracy": state["recent_accuracy"],
                    "rolling_accuracy": state["rolling_accuracy"],
                }
            return {
                "status": "pending",
                "assessment_ready": True,
                "session": session,
                "recent_accuracy": state["recent_accuracy"],
                "rolling_accuracy": state["rolling_accuracy"],
            }
        if (
            state["numeric_due"]
            and not bool(state["question_pool_complete"])
        ):
            checkpoint_reel_id = str(
                state.get("organizer_checkpoint_reel_id") or ""
            )
            self.prepare_reel_questions(
                conn,
                reel_ids=[checkpoint_reel_id] if checkpoint_reel_id else [
                    str(row.get("reel_id") or "")
                    for row in state["scroll_rows"]
                ],
                should_cancel=should_cancel,
                use_model=False,
            )
            state = self._readiness_state(conn, learner_id, material_id)
        if not state["assessment_ready"]:
            return {
                "status": "not_ready",
                "assessment_ready": False,
                "session": None,
                "recent_accuracy": state["recent_accuracy"],
                "rolling_accuracy": state["rolling_accuracy"],
            }
        count = self._desired_question_count(conn, state)
        selected = self._select_questions(state["available_questions"], count)
        question_target = max(
            1,
            int(state.get("session_question_target") or SESSION_QUESTION_TARGET),
        )
        if len(selected) != question_target:
            return {
                "status": "not_ready",
                "assessment_ready": bool(state["numeric_due"]),
                "session": None,
                "recent_accuracy": state["recent_accuracy"],
                "rolling_accuracy": state["rolling_accuracy"],
            }
        timestamp = now_iso()
        session_id = self._cadence_session_id(
            learner_id=learner_id,
            material_id=material_id,
            cadence_cutoff=str(state["cadence_cutoff"]),
        )
        _check_cancelled(should_cancel)
        with _atomic_write(conn):
            _check_cancelled(should_cancel)
            inserted = execute_modify(
                conn,
                """
                INSERT INTO assessment_sessions (
                    id, learner_id, material_id, status, current_index,
                    question_count, correct_count, information_units,
                    readiness_threshold, organizer_checkpoint_reel_id,
                    created_at, updated_at,
                    completed_at, snoozed_at
                ) VALUES (?, ?, ?, 'pending', 0, ?, 0, ?, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT DO NOTHING
                """,
                (
                    session_id,
                    learner_id,
                    material_id,
                    len(selected),
                    float(state["information_units"]),
                    float(state["readiness_threshold"]),
                    state.get("organizer_checkpoint_reel_id"),
                    timestamp,
                    timestamp,
                ),
            )
            if inserted <= 0:
                _check_cancelled(should_cancel)
                refreshed_state = self._readiness_state(
                    conn, learner_id, material_id
                )
                pending = refreshed_state["pending"]
                if pending:
                    return {
                        "status": "pending",
                        "assessment_ready": True,
                        "session": self._serialize_session(conn, pending),
                        "recent_accuracy": refreshed_state["recent_accuracy"],
                        "rolling_accuracy": refreshed_state["rolling_accuracy"],
                    }
                if (
                    str(refreshed_state["cadence_cutoff"])
                    != str(state["cadence_cutoff"])
                    or not refreshed_state["assessment_ready"]
                ):
                    return {
                        "status": "not_ready",
                        "assessment_ready": bool(
                            refreshed_state["assessment_ready"]
                        ),
                        "session": None,
                        "recent_accuracy": refreshed_state["recent_accuracy"],
                        "rolling_accuracy": refreshed_state["rolling_accuracy"],
                    }
                raise RuntimeError("assessment session conflict without a pending session")
            for position, question in enumerate(selected):
                _check_cancelled(should_cancel)
                execute_modify(
                    conn,
                    """
                    INSERT INTO assessment_session_questions (session_id, question_id, position)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, question["id"], position),
                )
            _check_cancelled(should_cancel)
        session = fetch_one(conn, "SELECT * FROM assessment_sessions WHERE id = ?", (session_id,))
        return {
            "status": "ready",
            "assessment_ready": True,
            "session": self._serialize_session(conn, session),
            "recent_accuracy": state["recent_accuracy"],
            "rolling_accuracy": state["rolling_accuracy"],
        }

    def _serialize_session(self, conn: Any, session: dict[str, Any] | None) -> dict[str, Any] | None:
        if not session:
            return None
        rows = fetch_all(
            conn,
            """
            SELECT
                sq.position,
                q.id,
                q.reel_id,
                q.prompt,
                q.options_json,
                r.concept_id,
                c.title AS concept_title,
                a.id AS attempt_id
            FROM assessment_session_questions sq
            JOIN reel_assessment_questions q ON q.id = sq.question_id
            JOIN reels r ON r.id = q.reel_id
            JOIN concepts c ON c.id = r.concept_id
            LEFT JOIN assessment_attempts a
              ON a.session_id = sq.session_id AND a.question_id = q.id
            WHERE sq.session_id = ?
            ORDER BY sq.position ASC
            """,
            (session["id"],),
        )
        questions: list[dict[str, Any]] = []
        first_unanswered = len(rows)
        answered_count = 0
        for row in rows:
            try:
                options = json.loads(str(row.get("options_json") or "[]"))
            except json.JSONDecodeError:
                options = []
            if row.get("attempt_id"):
                answered_count += 1
            elif first_unanswered == len(rows):
                first_unanswered = int(row.get("position") or 0)
            questions.append(
                {
                    "id": str(row["id"]),
                    "reel_id": str(row["reel_id"]),
                    "concept_id": str(row["concept_id"]),
                    "concept_title": str(row.get("concept_title") or ""),
                    "prompt": str(row.get("prompt") or ""),
                    "options": list(options) if isinstance(options, list) else [],
                }
            )
        status = str(session.get("status") or "pending")
        outcomes = fetch_all(
            conn,
            """
            SELECT o.adjustment, c.title AS concept_title
            FROM assessment_concept_outcomes o
            JOIN concepts c ON c.id = o.concept_id
            WHERE o.session_id = ?
            ORDER BY c.title ASC
            """,
            (session["id"],),
        )
        recent_accuracy, rolling_accuracy, _ = self._accuracy_stats(
            conn, str(session["learner_id"]), str(session["material_id"])
        )
        question_count = int(session.get("question_count") or len(rows))
        correct_count = int(session.get("correct_count") or 0)
        return {
            "id": str(session["id"]),
            "material_id": str(session["material_id"]),
            "status": status,
            "current_index": min(question_count, first_unanswered),
            "question_count": question_count,
            "answered_count": answered_count,
            "questions": questions,
            "score": (correct_count / max(1, question_count)) if status == "completed" else None,
            "understood_concepts": [
                str(row.get("concept_title") or "")
                for row in outcomes
                if float(row.get("adjustment") or 0.0) > 0
            ],
            "revisit_concepts": [
                str(row.get("concept_title") or "")
                for row in outcomes
                if float(row.get("adjustment") or 0.0) < 0
            ],
            "recent_accuracy": recent_accuracy,
            "rolling_accuracy": rolling_accuracy,
        }

    @staticmethod
    def _ensure_learner_progress(conn: Any, learner_id: str, material_id: str) -> None:
        if fetch_one(
            conn,
            "SELECT 1 AS present FROM learner_material_progress WHERE learner_id = ? AND material_id = ?",
            (learner_id, material_id),
        ):
            return
        material = fetch_one(conn, "SELECT knowledge_level FROM materials WHERE id = ?", (material_id,))
        timestamp = now_iso()
        upsert(
            conn,
            "learner_material_progress",
            {
                "learner_id": learner_id,
                "material_id": material_id,
                "selected_level": str((material or {}).get("knowledge_level") or "beginner"),
                "global_adjustment": 0.0,
                "difficulty_reset_at": timestamp,
                "feedback_revision": 0,
                "updated_at": timestamp,
            },
            pk=["learner_id", "material_id"],
        )

    @staticmethod
    def _maybe_auto_promote_level(
        conn: Any,
        *,
        learner_id: str,
        material_id: str,
        promoted_at: str,
    ) -> str | None:
        """Promote one named level after sustained, broad mastery evidence.

        Manual level changes reset ``difficulty_reset_at``, which naturally
        starts a fresh evidence window. Automatic demotion is intentionally not
        supported.
        """
        progress = fetch_one(
            conn,
            "SELECT selected_level, difficulty_reset_at FROM learner_material_progress "
            "WHERE learner_id = ? AND material_id = ?",
            (learner_id, material_id),
        ) or {}
        selected_level = str(progress.get("selected_level") or "beginner").strip().lower()
        next_level = {"beginner": "intermediate", "intermediate": "advanced"}.get(
            selected_level
        )
        if next_level is None:
            return None
        reset_at = str(progress.get("difficulty_reset_at") or "")
        outcomes = [
            row
            for row in fetch_all(
                conn,
                """
                SELECT o.session_id, o.concept_id, o.question_count,
                       o.correct_count, o.adjustment, o.created_at,
                       source_reel.search_context_json AS source_search_context_json
                FROM assessment_concept_outcomes o
                LEFT JOIN reels source_reel ON source_reel.id = o.source_reel_id
                WHERE o.learner_id = ? AND o.material_id = ? AND o.created_at > ?
                ORDER BY o.created_at DESC, o.session_id DESC, o.concept_id ASC
                """,
                (learner_id, material_id, reset_at),
            )
            if not has_incompatible_gemini_concept_family_contract(
                row.get("source_search_context_json")
            )
        ]
        total_questions = sum(max(0, int(row.get("question_count") or 0)) for row in outcomes)
        correct_questions = sum(max(0, int(row.get("correct_count") or 0)) for row in outcomes)
        concept_ids = {
            str(row.get("concept_id") or "").strip()
            for row in outcomes
            if str(row.get("concept_id") or "").strip()
        }
        if total_questions < 8 or len(concept_ids) < 3:
            return None
        if correct_questions / max(1, total_questions) < 0.85:
            return None

        if any(float(row.get("adjustment") or 0.0) < 0.0 for row in outcomes[:3]):
            return None

        execute_modify(
            conn,
            """
            UPDATE learner_material_progress
            SET selected_level = ?, global_adjustment = 0.0,
                difficulty_reset_at = ?, updated_at = ?
            WHERE learner_id = ? AND material_id = ?
            """,
            (next_level, promoted_at, promoted_at, learner_id, material_id),
        )
        return next_level

    def _finalize_session(self, conn: Any, session: dict[str, Any]) -> None:
        session_id = str(session["id"])
        self._ensure_learner_progress(
            conn, str(session["learner_id"]), str(session["material_id"])
        )
        timestamp = now_iso()
        transitioned = execute_modify(
            conn,
            """
            UPDATE assessment_sessions
            SET status = 'completed', current_index = question_count,
                completed_at = ?, updated_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (timestamp, timestamp, session_id),
        )
        if transitioned <= 0:
            return
        rows = fetch_all(
            conn,
            """
            SELECT
                r.concept_id,
                r.id AS reel_id,
                r.video_id,
                r.difficulty,
                a.is_correct,
                sq.position
            FROM assessment_session_questions sq
            JOIN reel_assessment_questions q ON q.id = sq.question_id
            JOIN reels r ON r.id = q.reel_id
            JOIN assessment_attempts a
              ON a.session_id = sq.session_id AND a.question_id = sq.question_id
            WHERE sq.session_id = ?
            ORDER BY sq.position ASC
            """,
            (session_id,),
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("concept_id") or ""), []).append(row)
        for concept_id, concept_rows in grouped.items():
            correct = sum(1 for row in concept_rows if int(row.get("is_correct") or 0) > 0)
            total = len(concept_rows)
            accuracy = correct / max(1, total)
            adjustment = 0.08 if accuracy >= 1.0 else -0.12 if accuracy < 0.5 else 0.0
            representative = next(
                (
                    row for row in concept_rows
                    if adjustment < 0 and int(row.get("is_correct") or 0) == 0
                ),
                concept_rows[0],
            )
            upsert(
                conn,
                "assessment_concept_outcomes",
                {
                    "learner_id": str(session["learner_id"]),
                    "session_id": session_id,
                    "material_id": str(session["material_id"]),
                    "concept_id": concept_id,
                    "question_count": total,
                    "correct_count": correct,
                    "accuracy": accuracy,
                    "adjustment": adjustment,
                    "source_reel_id": representative.get("reel_id"),
                    "source_video_id": representative.get("video_id"),
                    "source_difficulty": representative.get("difficulty"),
                    "created_at": timestamp,
                },
                pk=["session_id", "concept_id"],
            )
        self._maybe_auto_promote_level(
            conn,
            learner_id=str(session["learner_id"]),
            material_id=str(session["material_id"]),
            promoted_at=timestamp,
        )
        execute_modify(
            conn,
            """
            UPDATE learner_material_progress
            SET feedback_revision = feedback_revision + 1, updated_at = ?
            WHERE learner_id = ? AND material_id = ?
            """,
            (timestamp, session["learner_id"], session["material_id"]),
        )

    def answer(
        self,
        conn: Any,
        *,
        learner_id: str,
        session_id: str,
        question_id: str,
        choice_index: int,
    ) -> dict[str, Any]:
        if isinstance(choice_index, bool) or int(choice_index) < 0 or int(choice_index) > 3:
            raise ValueError("choice_index must be between 0 and 3")
        session = fetch_one(
            conn,
            "SELECT * FROM assessment_sessions WHERE id = ? AND learner_id = ?",
            (session_id, learner_id),
        )
        if not session:
            raise ValueError("assessment session not found")
        question = fetch_one(
            conn,
            """
            SELECT q.correct_index, q.explanation
            FROM assessment_session_questions sq
            JOIN reel_assessment_questions q ON q.id = sq.question_id
            WHERE sq.session_id = ? AND q.id = ?
            """,
            (session_id, question_id),
        )
        if not question:
            raise ValueError("assessment question not found")
        existing = fetch_one(
            conn,
            "SELECT * FROM assessment_attempts WHERE session_id = ? AND question_id = ?",
            (session_id, question_id),
        )
        if not existing:
            if str(session.get("status") or "") != "pending":
                raise ValueError("assessment session is not pending")
            state = self._readiness_state(
                conn,
                learner_id,
                str(session.get("material_id") or ""),
            )
            repaired = self._repair_pending_session(conn, state=state)
            if not self._exposable_pending_session(conn, repaired):
                if str(session.get("organizer_checkpoint_reel_id") or ""):
                    raise ValueError("assessment session is incomplete")
                raise ValueError(
                    "assessment session must contain at least three questions"
                )
            session = repaired or session
            expected = fetch_one(
                conn,
                """
                SELECT q.id AS question_id
                FROM assessment_session_questions sq
                JOIN reel_assessment_questions q ON q.id = sq.question_id
                LEFT JOIN assessment_attempts a
                  ON a.session_id = sq.session_id AND a.question_id = sq.question_id
                WHERE sq.session_id = ? AND a.id IS NULL
                ORDER BY sq.position ASC
                LIMIT 1
                """,
                (session_id,),
            )
            if not expected or str(expected.get("question_id") or "") != question_id:
                raise ValueError("assessment questions must be answered in order")
            correct_index = int(question["correct_index"])
            correct = int(choice_index) == correct_index
            timestamp = now_iso()
            with _atomic_write(conn):
                inserted = execute_modify(
                    conn,
                    """
                    INSERT INTO assessment_attempts (
                        id, learner_id, session_id, question_id,
                        choice_index, is_correct, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, question_id) DO NOTHING
                    """,
                    (
                        str(uuid.uuid4()),
                        learner_id,
                        session_id,
                        question_id,
                        int(choice_index),
                        1 if correct else 0,
                        timestamp,
                    ),
                )
                existing = fetch_one(
                    conn,
                    "SELECT * FROM assessment_attempts WHERE session_id = ? AND question_id = ?",
                    (session_id, question_id),
                )
                if not existing:
                    raise RuntimeError("assessment attempt conflict without a stored attempt")
                if inserted > 0:
                    totals = fetch_one(
                        conn,
                        """
                        SELECT COUNT(*) AS answered_count, COALESCE(SUM(is_correct), 0) AS correct_count
                        FROM assessment_attempts
                        WHERE session_id = ?
                        """,
                        (session_id,),
                    ) or {}
                    answered_count = int(totals.get("answered_count") or 0)
                    execute_modify(
                        conn,
                        """
                        UPDATE assessment_sessions
                        SET current_index = ?, correct_count = ?, updated_at = ?
                        WHERE id = ? AND status = 'pending'
                        """,
                        (
                            answered_count,
                            int(totals.get("correct_count") or 0),
                            timestamp,
                            session_id,
                        ),
                    )
                    if answered_count >= int(session.get("question_count") or 0):
                        session = fetch_one(
                            conn,
                            "SELECT * FROM assessment_sessions WHERE id = ?",
                            (session_id,),
                        ) or session
                        self._finalize_session(conn, session)
        session = fetch_one(conn, "SELECT * FROM assessment_sessions WHERE id = ?", (session_id,)) or session
        return {
            "correct": bool(int((existing or {}).get("is_correct") or 0)),
            "correct_index": int(question["correct_index"]),
            "explanation": str(question.get("explanation") or ""),
            "session": self._serialize_session(conn, session),
        }

    def snooze(self, conn: Any, *, learner_id: str, session_id: str) -> dict[str, Any]:
        session = fetch_one(
            conn,
            "SELECT * FROM assessment_sessions WHERE id = ? AND learner_id = ?",
            (session_id, learner_id),
        )
        if not session:
            raise ValueError("assessment session not found")
        status = str(session.get("status") or "")
        if status == "pending":
            timestamp = now_iso()
            execute_modify(
                conn,
                """
                UPDATE assessment_sessions
                SET status = 'snoozed', snoozed_at = ?, updated_at = ?
                WHERE id = ? AND learner_id = ? AND status = 'pending'
                """,
                (timestamp, timestamp, session_id, learner_id),
            )
            status = "snoozed"
        return {"status": status, "assessment_ready": False}

    def history_stats(
        self, conn: Any, *, learner_id: str, material_id: str
    ) -> dict[str, Any]:
        latest = fetch_one(
            conn,
            """
            SELECT correct_count, question_count, completed_at
            FROM assessment_sessions
            WHERE learner_id = ? AND material_id = ? AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 1
            """,
            (learner_id, material_id),
        )
        count_row = fetch_one(
            conn,
            """
            SELECT COUNT(*) AS completed_checks
            FROM assessment_sessions
            WHERE learner_id = ? AND material_id = ? AND status = 'completed'
            """,
            (learner_id, material_id),
        ) or {}
        recent, rolling, _ = self._accuracy_stats(conn, learner_id, material_id)
        return {
            "recent_correct": int((latest or {}).get("correct_count") or 0) if latest else None,
            "recent_total": int((latest or {}).get("question_count") or 0) if latest else None,
            "recent_accuracy": recent,
            "rolling_accuracy": rolling,
            "completed_checks": int(count_row.get("completed_checks") or 0),
            "last_completed_at": str((latest or {}).get("completed_at") or "") or None,
        }


__all__ = [
    "AssessmentCancelledError",
    "AssessmentService",
    "store_reel_assessment_question",
]
