"""Adaptive recall-check persistence, readiness, and session lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..db import (
    dumps_json,
    execute_modify,
    fetch_all,
    fetch_one,
    now_iso,
    upsert,
)
from ..clip_engine.errors import CancellationError, ProviderError


COMPLETION_FRACTION = 0.80
MIN_CADENCE_TARGET = 2
MAX_CADENCE_TARGET = 5
SESSION_QUESTION_TARGET = 2
CONCEPT_ADJUSTMENT_BOUND = 0.25
BACKFILL_CACHE_PREFIX = "assessment_question_backfill:"
NEGATIVE_BACKFILL_CACHE_TTL_SECONDS = 30

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
    def _cadence_target(
        *,
        learner_id: str,
        material_id: str,
        window_cutoff: str,
        recent_accuracy: float | None,
        scroll_rows: list[dict[str, Any]],
    ) -> int:
        """Choose a stable 2-5 reel cadence, then adapt it to current evidence."""
        seed = f"{learner_id}|{material_id}|{window_cutoff or 'initial'}"
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        target = MIN_CADENCE_TARGET + digest[0] % (
            MAX_CADENCE_TARGET - MIN_CADENCE_TARGET + 1
        )
        if recent_accuracy is not None:
            if float(recent_accuracy) < 0.70:
                target -= 1
            elif float(recent_accuracy) >= 0.90:
                target += 1
        if len(scroll_rows) >= 2:
            # Lock the continuity adjustment to the window's opening pair so
            # readiness cannot flap as later scroll events arrive.
            previous_concept = str(scroll_rows[0].get("concept_id") or "")
            current_concept = str(scroll_rows[1].get("concept_id") or "")
            if previous_concept and current_concept:
                target += 1 if previous_concept == current_concept else -1
        return max(MIN_CADENCE_TARGET, min(MAX_CADENCE_TARGET, target))

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
                q.prompt,
                q.options_json,
                q.created_at,
                r.concept_id,
                r.video_id,
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
        return [row for row in rows if len(row.get("options") or []) == 4]

    def _readiness_state(self, conn: Any, learner_id: str, material_id: str) -> dict[str, Any]:
        pending = self._pending_row(conn, learner_id, material_id)
        recent_accuracy, rolling_accuracy, recent_sessions = self._accuracy_stats(
            conn, learner_id, material_id
        )
        completed_cutoff = self._latest_cutoff(conn, learner_id, material_id, "completed")
        snoozed_cutoff = self._latest_cutoff(conn, learner_id, material_id, "snoozed")
        cadence_cutoff = max(completed_cutoff, snoozed_cutoff)
        completed_rows = self._completed_rows(conn, learner_id, material_id, completed_cutoff)
        total_units = self._information_units(completed_rows)
        scroll_rows = self._scrolled_rows(conn, learner_id, material_id, cadence_cutoff)
        cadence_target = self._cadence_target(
            learner_id=learner_id,
            material_id=material_id,
            window_cutoff=cadence_cutoff,
            recent_accuracy=recent_accuracy,
            scroll_rows=scroll_rows,
        )

        available = self._available_questions(conn, learner_id, material_id, cadence_cutoff)
        question_reels = {
            str(row.get("reel_id") or "")
            for row in available
            if str(row.get("reel_id") or "")
        }
        question_concepts = {
            str(row.get("concept_id") or "")
            for row in available
            if str(row.get("concept_id") or "")
        }
        scroll_concepts = {
            str(row.get("concept_id") or "")
            for row in scroll_rows
            if str(row.get("concept_id") or "")
        }
        question_target = min(SESSION_QUESTION_TARGET, len(scroll_concepts))
        question_pool_complete = (
            question_target > 0
            and len(question_reels) >= question_target
            and len(question_concepts) >= question_target
        )
        numeric_due = len(scroll_rows) >= cadence_target
        # Readiness is the durable due state. Question availability is handled
        # separately so a transient provider failure cannot erase that state.
        ready = bool(pending) or numeric_due
        return {
            "assessment_ready": ready,
            "numeric_due": numeric_due,
            "information_units": total_units,
            "readiness_threshold": float(cadence_target),
            "cadence_target": cadence_target,
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
                "Create one grounded four-option recall question for each supplied clip. "
                "Every option must be distinct. The explanation must be supported by that clip's transcript. "
                "Keep each prompt at most 16 words, each option at most 8 words, and each "
                "explanation one sentence and at most 24 words. "
                "Return JSON only as {\"questions\":[{\"reel_id\":...,\"prompt\":...,"
                "\"options\":[... exactly four ...],\"correct_index\":0-3,\"explanation\":...}]}"
            ),
            dumps_json({"clips": clips}),
            _BackfillPlan,
            temperature=0.2,
            model=clipper_config.SEGMENT_MODEL,
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
    ) -> None:
        del learner_id, material_id  # content is reel-scoped; eligibility is learner-scoped.
        uncached: list[dict[str, Any]] = []
        satisfied_reels: set[str] = set()
        satisfied_concepts: set[str] = set()
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
        question_target = min(SESSION_QUESTION_TARGET, len(seen_concepts))

        def pool_complete() -> bool:
            return (
                question_target > 0
                and len(satisfied_reels) >= question_target
                and len(satisfied_concepts) >= question_target
            )

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
                continue
            candidates.append({**row, "fingerprint": fingerprint})
        if pool_complete():
            return

        candidates.sort(
            key=lambda row: (
                int(str(row.get("concept_id") or "") not in satisfied_concepts),
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
                        if pool_complete():
                            return
                continue
            uncached.append(row)
            projected_reels = satisfied_reels | {
                str(candidate.get("reel_id") or "") for candidate in uncached
            }
            projected_concepts = satisfied_concepts | {
                str(candidate.get("concept_id") or "") for candidate in uncached
            }
            if (
                len(projected_reels) >= question_target
                and len(projected_concepts) >= question_target
            ):
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
            self._write_cached_backfill(conn, str(row["fingerprint"]), cached_value)

    def _desired_question_count(self, conn: Any, state: dict[str, Any]) -> int:
        del conn
        distinct_reels = {
            str(row.get("reel_id") or "")
            for row in state["available_questions"]
            if str(row.get("reel_id") or "")
        }
        distinct_concepts = {
            str(row.get("concept_id") or "")
            for row in state["available_questions"]
            if str(row.get("concept_id") or "")
        }
        return min(
            SESSION_QUESTION_TARGET,
            len(distinct_reels),
            len(distinct_concepts),
        )

    @staticmethod
    def _select_questions(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        remaining = list(rows)
        chosen: list[dict[str, Any]] = []
        concepts: set[str] = set()
        reels: set[str] = set()
        videos: set[str] = set()
        while remaining and len(chosen) < count:
            distinct_rows = [
                row
                for row in remaining
                if str(row.get("reel_id") or "") not in reels
                and str(row.get("concept_id") or "") not in concepts
            ]
            if not distinct_rows:
                break
            best = max(
                distinct_rows,
                key=lambda row: (
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

    def pending(self, conn: Any, *, learner_id: str, material_id: str) -> dict[str, Any]:
        state = self._readiness_state(conn, learner_id, material_id)
        session = self._serialize_session(conn, state["pending"]) if state["pending"] else None
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
            return {
                "status": "pending",
                "assessment_ready": True,
                "session": self._serialize_session(conn, state["pending"]),
                "recent_accuracy": state["recent_accuracy"],
                "rolling_accuracy": state["rolling_accuracy"],
            }
        if (
            state["numeric_due"]
            and not bool(state["question_pool_complete"])
        ):
            self._ensure_question_pool(
                conn,
                learner_id=learner_id,
                material_id=material_id,
                source_rows=state["scroll_rows"],
                should_cancel=should_cancel,
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
        if not selected:
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
                    readiness_threshold, created_at, updated_at,
                    completed_at, snoozed_at
                ) VALUES (?, ?, ?, 'pending', 0, ?, 0, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT DO NOTHING
                """,
                (
                    session_id,
                    learner_id,
                    material_id,
                    len(selected),
                    float(state["information_units"]),
                    float(state["readiness_threshold"]),
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
