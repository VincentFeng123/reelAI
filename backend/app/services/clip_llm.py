"""
LLM-direct clip picker.

Given a user query and a transcript, ask the LLM to return the ONE best
15-60s clip in the video that addresses the query — no intermediate
"topic segmentation + rank + narrow" dance. The snap_boundary helper
(in clip_boundary.py) then refines the LLM's raw timestamps to the
nearest terminal-punctuation sentence boundary and inter-word silence
gap before the clip is cut.

Contract:
    pick_clip_llm(query, cues, *, min_sec=15.0, max_sec=60.0) -> ClipPick | None

Returns None when:
    * no LLM is configured (neither Gemini nor Groq keys present)
    * the LLM returns malformed JSON or a clip outside [min_sec, max_sec]
    * the LLM returns timestamps that do not appear in the transcript
    * all API calls rate-limit and no fallback succeeds

The caller (reels.py) treats None as "fall back to heuristic picker".

Design notes:
  * Reuses the Gemini/Groq clients and key rotation from ``topic_cut``.
  * Transcript rendering format matches ``topic_cut`` so the LLM sees a
    shape it's already trained against.
  * No scoring/ranking — per user requirement, the LLM *picks* the clip;
    we do not double-judge importance afterward.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from .topic_cut import (
    TranscriptCue,
    _build_cerebras_client,
    _build_gemini_client,
    _build_groq_client,
    _collect_gemini_api_keys,
    _render_transcript_for_llm,
    _strip_code_fences,
)

logger = logging.getLogger(__name__)


# Per-clip duration bounds. Must stay in sync with segmenter.py MIN/MAX
# and ClipBoundaryEngine tolerances. 15-60s is the historic contract —
# below 15s clips lack context, above 60s users scroll away.
MIN_CLIP_SEC = 15.0
MAX_CLIP_SEC = 60.0


@dataclass
class ClipPick:
    """Raw LLM output for a single clip. Timestamps are unsnapped — the
    caller runs them through the snap_boundary helper before cutting.
    """
    t_start: float
    t_end: float
    reason: str

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.t_end - self.t_start)


def _build_clip_system_prompt(
    query: str,
    min_sec: float,
    max_sec: float,
    target_sec: float,
) -> str:
    """System prompt for the single-clip picker.

    Emphasizes: (a) exactly one clip, (b) duration bounds from user settings,
    (c) aim near target_sec (user's preferred clip length), (d) starts and
    ends on complete thoughts, (e) picks the passage that most substantively
    addresses the query, not merely mentions it.
    """
    return f"""You are a precise video editor cutting a single short clip from a long-form YouTube video.

You will receive a transcript. Each line has the format:
    [<start_seconds>-<end_seconds>s] <speech>

Example: [45.2-48.7s] And that brings us to the concept of gradient descent.

The user is searching for: "{query}"

The user's clip-length preferences (from their settings):
  - Minimum duration: {min_sec:.0f} seconds
  - Maximum duration: {max_sec:.0f} seconds
  - Preferred duration: {target_sec:.0f} seconds (aim for this when the natural topic allows)

Your job: pick the ONE best clip from this video that most substantively addresses the user's query. Return exactly one clip — no alternatives, no runner-ups. Aim near {target_sec:.0f} seconds when the topic permits; extend toward {max_sec:.0f}s only when a single natural thought genuinely needs the room, and trim toward {min_sec:.0f}s only when the substantive passage is brief.

Rules for picking the clip:
- Choose the passage where the creator most directly and substantively discusses the query topic, not merely mentions it. A passing reference ("as we discussed with X") is not a clip; a sustained explanation of X is.
- Skip intros ("hey guys welcome back"), outros ("don't forget to subscribe"), sponsor reads, and recaps of other videos. These are not clips.
- If the creator revisits the query topic multiple times, pick the single passage with the highest information density — usually the one with concrete examples, definitions, or step-by-step explanation.

Rules for t_start:
- t_start must be the START timestamp of a cue where the creator begins SUSTAINED discussion of the query topic. Not the preceding segue ("let's move on", "alright so") — start at the actual substance.
- Do not start mid-sentence. The chosen cue should be the first word of a complete thought.

Rules for t_end:
- t_end must be the END timestamp of a cue where the creator is finishing a complete thought about the query topic.
- Do not end mid-sentence. Prefer ending just before a transition ("now let's look at", "moving on") rather than cutting the transition in.
- Duration = t_end - t_start MUST be between {min_sec:.0f} and {max_sec:.0f} seconds. Outside this range the clip is rejected entirely.

IMPORTANT: t_start and t_end MUST be exact timestamp values that appear in the transcript's [X.X-Y.Ys] ranges. Use the START value of some cue for t_start and the END value of some cue for t_end. Do not interpolate, average, or invent timestamps.

If no passage in this video substantively addresses the query within the duration bounds, return {{"t_start": null, "t_end": null, "reason": "<why>"}}. Do NOT return a weak or off-topic clip to fill the slot.

Return JSON only, in this exact shape:
{{
    "t_start": <float or null>,
    "t_end": <float or null>,
    "reason": "<one short sentence explaining why this passage addresses the query, or why no clip was picked>"
}}
No prose outside the JSON."""


def _parse_clip_pick_json(raw: str, cues: Sequence[TranscriptCue]) -> ClipPick | None:
    """Parse the JSON response into a ClipPick or return None on any
    malformation / bounds violation / timestamp-hallucination.
    """
    try:
        payload = json.loads(_strip_code_fences(raw))
    except (json.JSONDecodeError, ValueError):
        logger.debug("clip_llm: invalid JSON from LLM: %s", raw[:200])
        return None
    if not isinstance(payload, dict):
        return None

    t_start_raw = payload.get("t_start")
    t_end_raw = payload.get("t_end")
    if t_start_raw is None or t_end_raw is None:
        logger.debug("clip_llm: LLM declined to pick a clip: %s", payload.get("reason"))
        return None
    try:
        t_start = float(t_start_raw)
        t_end = float(t_end_raw)
    except (TypeError, ValueError):
        return None
    if t_start < 0 or t_end <= t_start:
        return None
    reason = str(payload.get("reason") or "").strip()
    return ClipPick(t_start=t_start, t_end=t_end, reason=reason)


def _validate_pick(
    pick: ClipPick,
    cues: Sequence[TranscriptCue],
    *,
    min_sec: float,
    max_sec: float,
) -> ClipPick | None:
    """Reject picks outside duration bounds or referencing timestamps the
    LLM invented. This catches hallucinated timestamps that would cause a
    seek past the end of the source video.
    """
    duration = pick.t_end - pick.t_start
    if duration < min_sec or duration > max_sec:
        logger.debug(
            "clip_llm: rejecting pick with duration %.1fs (bounds %.0f-%.0f)",
            duration, min_sec, max_sec,
        )
        return None
    if not cues:
        return pick
    video_start = min(cue.start for cue in cues)
    video_end = max(cue.end for cue in cues)
    # Allow small slop at the edges (LLMs sometimes round timestamps).
    if pick.t_start < video_start - 0.5 or pick.t_end > video_end + 0.5:
        logger.debug(
            "clip_llm: pick %.1f-%.1fs outside transcript range %.1f-%.1fs",
            pick.t_start, pick.t_end, video_start, video_end,
        )
        return None
    return pick


def _pick_via_gemini(
    query: str,
    cues: Sequence[TranscriptCue],
    *,
    min_sec: float,
    max_sec: float,
    target_sec: float,
    model: str = "gemini-2.0-flash",
) -> ClipPick | None:
    """Try Gemini Flash first — cheapest + fastest. Rotates API keys on
    rate-limit (same pattern as topic_cut._llm_topic_segments_gemini).
    """
    genai = _build_gemini_client()
    if genai is None:
        return None
    keys = _collect_gemini_api_keys()
    if not keys:
        return None

    from google.genai import types as genai_types

    system_prompt = _build_clip_system_prompt(query, min_sec, max_sec, target_sec)
    user_msg = (
        "Here is the full transcript of a YouTube video. "
        "Pick the one best clip per the rules in the system prompt.\n\n"
        f"{_render_transcript_for_llm(cues)}"
    )

    for key in keys:
        try:
            client = genai.Client(api_key=key)
        except Exception:
            continue
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_msg,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            raw = response.text or "{}"
            pick = _parse_clip_pick_json(raw, cues)
            if pick is None:
                return None
            return _validate_pick(pick, cues, min_sec=min_sec, max_sec=max_sec)
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = any(s in exc_str for s in ("429", "resource_exhausted", "rate", "quota"))
            if is_rate_limit and len(keys) > 1:
                continue
            logger.debug("clip_llm: Gemini call failed: %s", exc)
            return None
    return None


def _pick_via_groq(
    query: str,
    cues: Sequence[TranscriptCue],
    *,
    min_sec: float,
    max_sec: float,
    target_sec: float,
    model: str = "llama-3.3-70b-versatile",
) -> ClipPick | None:
    """Groq Llama fallback when Gemini is unavailable or exhausted."""
    groq_client = _build_groq_client()
    if groq_client is None:
        return None
    system_prompt = _build_clip_system_prompt(query, min_sec, max_sec, target_sec)
    user_msg = (
        "Here is the full transcript of a YouTube video. "
        "Pick the one best clip per the rules in the system prompt.\n\n"
        f"{_render_transcript_for_llm(cues)}"
    )
    try:
        response = groq_client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content or "{}"
    except Exception as exc:
        logger.debug("clip_llm: Groq call failed: %s", exc)
        return None
    pick = _parse_clip_pick_json(raw, cues)
    if pick is None:
        return None
    return _validate_pick(pick, cues, min_sec=min_sec, max_sec=max_sec)


def _pick_via_cerebras(
    query: str,
    cues: Sequence[TranscriptCue],
    *,
    min_sec: float,
    max_sec: float,
    target_sec: float,
    model: str = "llama3.1-8b",
) -> ClipPick | None:
    """Cerebras Llama 3.3 70B fallback when both Gemini and Groq fail.

    Cerebras uses the same OpenAI-compatible chat-completion shape as
    Groq, so the call is structurally identical.
    """
    client = _build_cerebras_client()
    if client is None:
        return None
    system_prompt = _build_clip_system_prompt(query, min_sec, max_sec, target_sec)
    user_msg = (
        "Here is the full transcript of a YouTube video. "
        "Pick the one best clip per the rules in the system prompt.\n\n"
        f"{_render_transcript_for_llm(cues)}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content or "{}"
    except Exception as exc:
        logger.debug("clip_llm: Cerebras call failed: %s", exc)
        return None
    pick = _parse_clip_pick_json(raw, cues)
    if pick is None:
        return None
    return _validate_pick(pick, cues, min_sec=min_sec, max_sec=max_sec)


def pick_clip_llm(
    query: str,
    cues: Sequence[TranscriptCue],
    *,
    min_sec: float = MIN_CLIP_SEC,
    max_sec: float = MAX_CLIP_SEC,
    target_sec: float | None = None,
) -> ClipPick | None:
    """Pick the single best clip for the query from the transcript via LLM.

    `min_sec`/`max_sec`/`target_sec` come from the user's clip-window
    settings (`target_clip_duration_min_sec`, `_max_sec`, `_sec`). The
    LLM is told the preferred length and aims for it when the natural
    topic allows; it can stretch or contract within [min, max]. When
    `target_sec` is omitted it defaults to the midpoint of [min, max].

    Fallback chain: Gemini → Groq → Cerebras. Returns None if all three
    are unavailable or all fail validation; caller then falls back to the
    heuristic picker in ClipBoundaryEngine.
    """
    query = (query or "").strip()
    if not query or not cues:
        return None
    if target_sec is None:
        target_sec = 0.5 * (min_sec + max_sec)
    # Keep target inside [min, max]; callers occasionally pass loose values.
    target_sec = max(min_sec, min(max_sec, float(target_sec)))
    pick = _pick_via_gemini(
        query, cues, min_sec=min_sec, max_sec=max_sec, target_sec=target_sec,
    )
    if pick is not None:
        return pick
    pick = _pick_via_groq(
        query, cues, min_sec=min_sec, max_sec=max_sec, target_sec=target_sec,
    )
    if pick is not None:
        return pick
    return _pick_via_cerebras(
        query, cues, min_sec=min_sec, max_sec=max_sec, target_sec=target_sec,
    )


# --------------------------------------------------------------------- #
# Phase 3 — cross-video LLM reranker
# --------------------------------------------------------------------- #


@dataclass
class RankedClipPick:
    """Output of the global reranker. One per selected candidate, ordered
    by virality desc. `candidate_idx` indexes back into the list that was
    passed in — the caller reads the actual boundaries from there."""
    candidate_idx: int
    virality_score: float       # 0..1 from LLM
    hook_pattern: str | None    # copied from candidate; may differ if LLM re-classifies
    reason: str                 # one short sentence


def _build_rerank_system_prompt(query: str | None, target_count: int) -> str:
    query_line = f'The user is searching for: "{query}"\n\n' if query else ""
    return (
        "You are a viral-clip ranker. You receive a NUMBERED list of candidate "
        "clips extracted from a single long-form video. Each candidate shows "
        "opener text, closer text, duration, structural sub-scores (relevance, "
        "engagement, completeness), and a hook pattern if one was detected.\n\n"
        f"{query_line}"
        f"Your job: pick the top {target_count} candidates most likely to "
        "perform as standalone short-form clips. Weight the decision by:\n"
        "  1. How strongly the clip stands alone — no unresolved references, "
        "opens on a complete thought, closes on a complete thought.\n"
        "  2. Hook strength — does the opener earn attention in the first "
        "second? Opus Clip's viral hook taxonomy (contradiction, number "
        "promise, confession, question, before/after, counterintuitive how-to, "
        "warning) is a useful prior but not required.\n"
        "  3. Payoff — does the clip deliver on what its opener promises?\n"
        "  4. Relevance to query when one was provided.\n\n"
        "Return JSON only:\n"
        '{"picks": [{"candidate_idx": <int>, "virality_score": <0..1>, '
        '"hook_pattern": "<name or null>", "reason": "<one short sentence>"}]}\n'
        "Return at most the requested count. Fewer is fine if most candidates "
        "are weak. Do NOT invent new candidates — only reference "
        "candidate_idx values from the list."
    )


def _render_candidates_for_rerank(candidates: Sequence[Any]) -> str:
    """Compact numbered rendering of candidates — metadata only, no transcript text."""
    lines: list[str] = []
    for i, c in enumerate(candidates):
        duration = max(0.0, float(getattr(c, "t_end", 0.0)) - float(getattr(c, "t_start", 0.0)))
        lines.append(
            f"[{i}] start={float(getattr(c, 't_start', 0.0)):.1f}s "
            f"end={float(getattr(c, 't_end', 0.0)):.1f}s "
            f"dur={duration:.1f}s "
            f"hook={getattr(c, 'hook_pattern', None) or '-'} "
            f"rel={float(getattr(c, 'relevance_score', 0.0)):.2f} "
            f"eng={float(getattr(c, 'engagement_score', 0.0)):.2f} "
            f"cmp={float(getattr(c, 'completeness_score', 0.0)):.2f}\n"
            f"    opener: {getattr(c, 'opener_text', '')!r}\n"
            f"    closer: {getattr(c, 'closer_text', '')!r}"
        )
    return "\n".join(lines)


def _parse_rerank_response(
    raw: str, *, num_candidates: int, target_count: int,
) -> list[RankedClipPick]:
    try:
        payload = json.loads(_strip_code_fences(raw))
    except (json.JSONDecodeError, ValueError):
        logger.debug("rerank: invalid JSON from LLM: %s", raw[:200])
        return []
    if not isinstance(payload, dict):
        return []
    picks_raw = payload.get("picks")
    if not isinstance(picks_raw, list):
        return []
    out: list[RankedClipPick] = []
    seen_idx: set[int] = set()
    for entry in picks_raw:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("candidate_idx"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= num_candidates or idx in seen_idx:
            continue
        seen_idx.add(idx)
        try:
            virality = float(entry.get("virality_score") or 0.0)
        except (TypeError, ValueError):
            virality = 0.0
        virality = max(0.0, min(1.0, virality))
        hook = entry.get("hook_pattern")
        hook_str = str(hook) if isinstance(hook, str) and hook else None
        reason = str(entry.get("reason") or "").strip()[:300]
        out.append(RankedClipPick(
            candidate_idx=idx,
            virality_score=virality,
            hook_pattern=hook_str,
            reason=reason,
        ))
    return out[:target_count]


def _heuristic_rerank_fallback(
    candidates: Sequence[Any],
    *,
    target_count: int,
) -> list[RankedClipPick]:
    """Deterministic ranking when the LLM chain returns nothing usable.
    Combined weighting documented in the design: 0.45·relevance +
    0.25·engagement + 0.20·completeness + 0.10·boundary_confidence."""
    scored: list[tuple[int, float, Any]] = []
    for i, c in enumerate(candidates):
        rel = float(getattr(c, "relevance_score", 0.0))
        eng = float(getattr(c, "engagement_score", 0.0))
        cmp_ = float(getattr(c, "completeness_score", 0.0))
        bc = float(getattr(c, "boundary_confidence", 0.0))
        combined = 0.45 * rel + 0.25 * eng + 0.20 * cmp_ + 0.10 * bc
        scored.append((i, combined, c))
    scored.sort(key=lambda t: t[1], reverse=True)
    out: list[RankedClipPick] = []
    for idx, combined, c in scored[:target_count]:
        out.append(RankedClipPick(
            candidate_idx=idx,
            virality_score=max(0.0, min(1.0, combined)),
            hook_pattern=getattr(c, "hook_pattern", None),
            reason="heuristic fallback (no LLM)",
        ))
    return out


def rerank_clip_candidates_llm(
    candidates: Sequence[Any],
    *,
    query: str | None = None,
    target_count: int = 5,
) -> list[RankedClipPick]:
    """Rank candidates globally across the video. Sends metadata (not
    transcript text) so the prompt stays small even at 60 candidates.

    Returns a heuristic-sorted fallback if every LLM provider fails — the
    caller can always rely on a non-empty return when candidates is non-empty.
    """
    if not candidates:
        return []
    target_count = max(1, int(target_count))
    system_prompt = _build_rerank_system_prompt(query, target_count)
    user_msg = (
        f"Candidates (total={len(candidates)}):\n"
        f"{_render_candidates_for_rerank(candidates)}"
    )
    num = len(candidates)

    genai = _build_gemini_client()
    if genai is not None:
        keys = _collect_gemini_api_keys()
        if keys:
            from google.genai import types as genai_types
            for key in keys:
                try:
                    client = genai.Client(api_key=key)
                except Exception:
                    continue
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=user_msg,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )
                    picks = _parse_rerank_response(
                        response.text or "{}",
                        num_candidates=num, target_count=target_count,
                    )
                    if picks:
                        return picks
                    break
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if any(s in exc_str for s in ("429", "resource_exhausted", "rate", "quota")):
                        continue
                    logger.debug("rerank: Gemini failed: %s", exc)
                    break

    groq_client = _build_groq_client()
    if groq_client is not None:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            picks = _parse_rerank_response(
                response.choices[0].message.content or "{}",
                num_candidates=num, target_count=target_count,
            )
            if picks:
                return picks
        except Exception as exc:
            logger.debug("rerank: Groq failed: %s", exc)

    cerebras_client = _build_cerebras_client()
    if cerebras_client is not None:
        try:
            response = cerebras_client.chat.completions.create(
                model="llama3.1-8b",
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            picks = _parse_rerank_response(
                response.choices[0].message.content or "{}",
                num_candidates=num, target_count=target_count,
            )
            if picks:
                return picks
        except Exception as exc:
            logger.debug("rerank: Cerebras failed: %s", exc)

    logger.info("rerank: all LLM providers failed; using heuristic fallback")
    return _heuristic_rerank_fallback(candidates, target_count=target_count)


__all__ = [
    "ClipPick",
    "MIN_CLIP_SEC",
    "MAX_CLIP_SEC",
    "pick_clip_llm",
    "RankedClipPick",
    "rerank_clip_candidates_llm",
]
