"""
ClipBoundaryEngine — precision start/end picking for clips (Phase A.3).

Contract:
  - The LLM and chapter logic in `topic_cut.py` decide WHICH topic window
    to extract from a long-form video.
  - This engine decides the EXACT frame to start and end the clip at,
    working off word-level transcript data (Phase A.1) and sentence spans
    (Phase A.2).

Behavior:
  - `find_topic_start`: pick the first substantive-introduction sentence
    within a candidate window. Uses LLM when available (short prompt over
    the first ~8 sentences), falls back to a lexical-novelty heuristic:
    first sentence with ≥ MIN_NOVELTY_JACCARD distance from the 30s of
    material preceding the window AND ≥1 topic keyword. Snaps to the
    `word_start_idx` of the chosen sentence — not cue start.

  - `find_topic_end`: two modes.
      NATURAL CLOSE: if any `topic_switch_hint` lies within
        [user_min * 0.9, user_max * 1.25], pick the last sentence ending
        strictly before the hint. The 1.25x tolerance reflects the user's
        intent to treat min/max as soft references, not hard caps.
      TRUNCATION: if no hint lands in range, pick the latest sentence
        ending at or before `user_max * 1.1` with a terminal-punct char
        {'.', '!', '?', '…'}. Never end on a comma, never mid-word.
      Tiebreaker for both modes: prefer sentences whose next 350ms contains
      silence (inhale/pause = clean break), when silence ranges are available.

  - `detect_topic_switches`: unify chapter boundaries (confidence 1.0),
    LLM-provided switch hints (~0.7), and lexical-novelty spikes (~0.4)
    into a single ordered list.

  - `refine`: the public entrypoint. Given a raw `(t_start, t_end)` from
    topic_cut's LLM/chapter/novelty decisions, snap both ends via the above
    primitives.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..ingestion.models import IngestTranscriptCue
from .sentences import (
    SentenceSpan,
    first_sentence_at_or_after,
    latest_sentence_ending_before,
    sentences_in_range,
    split_sentences,
)

logger = logging.getLogger(__name__)

BOUNDARY_ENGINE_VERSION = 2

# How far past user_max we tolerate to land a clean close. 1.25x means a
# 55s target can stretch to ~68s when the topic naturally ends there.
SOFT_MAX_TOLERANCE = 1.25
# How far past user_max truncation mode will reach to find a terminal-punct
# sentence when no topic-switch hint exists. Tighter than SOFT_MAX_TOLERANCE
# because we're truncating, not closing naturally.
TRUNCATION_TOLERANCE = 1.10
# Lower bound for "natural close" — don't take a close that's << user_min.
SOFT_MIN_TOLERANCE = 0.90

# Sentence introducing a topic must differ from prior context by at least this Jaccard distance.
MIN_NOVELTY_JACCARD = 0.35
# Seconds of prior material to compare against for the novelty check.
PRIOR_CONTEXT_SEC = 30.0
# How close a silence-range start must be to a sentence end to count as a tiebreaker anchor.
SILENCE_TIEBREAK_WINDOW_SEC = 0.35

# Keyword extraction for lexical checks.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by",
        "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
        "his", "her", "its", "our", "their", "not", "no", "so", "if", "then",
        "than", "too", "very", "just", "can", "will", "would", "could", "should",
        "like", "going", "get", "got", "ok", "okay", "right", "yeah", "well",
        "now", "some", "any", "all", "one", "two", "there", "here", "what",
        "how", "why", "when", "where", "who", "whom", "whose",
    }
)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS and len(t) > 2}


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return 1.0 - (inter / union)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass
class TopicSwitchHint:
    t_sec: float
    confidence: float  # 0.0-1.0
    source: str  # "chapter" | "llm" | "novelty" | "silence"
    label: str = ""


@dataclass
class BoundaryResult:
    t_start: float
    t_end: float
    start_sentence: SentenceSpan | None
    end_sentence: SentenceSpan | None
    start_mode: str  # "llm" | "novelty" | "fallback"
    end_mode: str  # "natural-close" | "truncation" | "fallback"
    boundary_quality: str  # "precise" | "sentence" | "chapter" | "degraded"
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Chapter input shape
# --------------------------------------------------------------------------- #


def _chapters_to_hints(chapters: Sequence[dict[str, Any]] | None) -> list[TopicSwitchHint]:
    """
    Convert yt-dlp-style chapters (`[{"start_time", "end_time", "title"}]`) into
    switch hints at each chapter's `start_time` (the BOUNDARY is where topic N+1
    begins). We emit a hint at the start of every chapter AFTER the first.
    """
    hints: list[TopicSwitchHint] = []
    if not chapters:
        return hints
    sorted_chaps = sorted(
        [c for c in chapters if isinstance(c, dict) and isinstance(c.get("start_time"), (int, float))],
        key=lambda c: float(c.get("start_time") or 0.0),
    )
    for i, chap in enumerate(sorted_chaps):
        if i == 0:
            continue
        try:
            t = float(chap.get("start_time") or 0.0)
        except (TypeError, ValueError):
            continue
        title = str(chap.get("title") or "")
        hints.append(TopicSwitchHint(t_sec=t, confidence=1.0, source="chapter", label=title))
    return hints


def _llm_hints_to_switch_hints(llm_raw: Iterable[dict[str, Any]] | None) -> list[TopicSwitchHint]:
    """Convert `topic_cut.py`-style topic segments into end-of-topic switch hints."""
    if not llm_raw:
        return []
    out: list[TopicSwitchHint] = []
    for seg in llm_raw:
        try:
            t_end = float(seg.get("t_end") if isinstance(seg, dict) else getattr(seg, "t_end", 0.0))
        except (TypeError, ValueError):
            continue
        if t_end <= 0:
            continue
        label = str((seg.get("label") if isinstance(seg, dict) else getattr(seg, "label", "")) or "")
        out.append(TopicSwitchHint(t_sec=t_end, confidence=0.7, source="llm", label=label))
    return out


# --------------------------------------------------------------------------- #
# Lexical-novelty spike detection (fallback topic-switch signal)
# --------------------------------------------------------------------------- #


def _novelty_hints_from_cues(
    cues: list[IngestTranscriptCue],
    *,
    window_sec: float = 30.0,
    min_spike: float = 0.60,
) -> list[TopicSwitchHint]:
    """
    Rolling Jaccard-distance on 30s windows; spikes above `min_spike` are
    candidate topic switches. Low confidence since this is purely lexical.
    """
    if not cues or len(cues) < 6:
        return []
    # Build cumulative token sets per cue for a linear pass.
    per_cue_tokens: list[set[str]] = [_tokenize(c.text) for c in cues]
    per_cue_end: list[float] = [c.end for c in cues]
    hints: list[TopicSwitchHint] = []
    # Slide: compare tokens in [t - window, t] vs [t, t + window/2].
    for i in range(2, len(cues) - 2):
        t = per_cue_end[i]
        left_lo = t - window_sec
        right_hi = t + window_sec * 0.5
        left: set[str] = set()
        right: set[str] = set()
        for j, c in enumerate(cues):
            if c.end < left_lo:
                continue
            if c.start > right_hi:
                break
            if c.end <= t:
                left |= per_cue_tokens[j]
            else:
                right |= per_cue_tokens[j]
        dist = _jaccard_distance(left, right)
        if dist >= min_spike:
            hints.append(TopicSwitchHint(t_sec=t, confidence=min(0.45, 0.3 + (dist - min_spike)), source="novelty"))
    # Dedupe: keep only local maxima within a 5s window.
    if not hints:
        return hints
    hints.sort(key=lambda h: h.t_sec)
    pruned: list[TopicSwitchHint] = []
    for h in hints:
        if pruned and (h.t_sec - pruned[-1].t_sec) < 5.0:
            if h.confidence > pruned[-1].confidence:
                pruned[-1] = h
            continue
        pruned.append(h)
    return pruned


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class ClipBoundaryEngine:
    """
    Start/end boundary picker. LLM-call optional — pass `llm_pick_start=None`
    to disable and rely purely on the novelty heuristic.
    """

    def __init__(
        self,
        *,
        llm_pick_start: Callable[[str, list[SentenceSpan]], int | None] | None = None,
    ) -> None:
        # `llm_pick_start(query, candidate_sentences)` returns the index of the
        # first substantive introduction, or None to defer to the heuristic.
        self._llm_pick_start = llm_pick_start

    # ---- public API ------------------------------------------------------- #

    def refine(
        self,
        *,
        raw_t_start: float,
        raw_t_end: float,
        cues: list[IngestTranscriptCue],
        sentences: list[SentenceSpan] | None = None,
        query: str | None = None,
        chapters: Sequence[dict[str, Any]] | None = None,
        llm_topic_segments: Iterable[dict[str, Any]] | None = None,
        silence_ranges: Sequence[tuple[float, float]] | None = None,
        user_min_sec: float = 20.0,
        user_max_sec: float = 55.0,
        video_duration_sec: float | None = None,
    ) -> BoundaryResult:
        """
        Snap a raw LLM/chapter-supplied (t_start, t_end) to precise sentence+word
        boundaries. See module docstring for the rules.
        """
        warnings: list[str] = []
        if sentences is None:
            sentences = split_sentences(cues)
        if not sentences:
            return BoundaryResult(
                t_start=raw_t_start,
                t_end=raw_t_end,
                start_sentence=None,
                end_sentence=None,
                start_mode="fallback",
                end_mode="fallback",
                boundary_quality="degraded",
                warnings=["no_sentences_available"],
            )

        hints = self._build_switch_hints(cues, chapters, llm_topic_segments, silence_ranges)

        # ---- START ---- #
        start_window_hi = min(raw_t_start + max(user_max_sec * 0.25, 8.0), raw_t_end)
        window_sents = sentences_in_range(sentences, raw_t_start, start_window_hi)
        start_sent, start_mode = self._pick_topic_start(
            sentences_all=sentences,
            window_sents=window_sents,
            raw_t_start=raw_t_start,
            query=query,
            silence_ranges=silence_ranges,
        )
        if start_sent is None:
            # Degrade: use the first sentence at or after raw_t_start.
            start_sent = first_sentence_at_or_after(sentences, raw_t_start)
            start_mode = "fallback"
        if start_sent is None:
            warnings.append("no_start_sentence_found")
            start_t = raw_t_start
        else:
            start_t = start_sent.t_start

        # ---- END ---- #
        # Hints considered are those strictly after start_t.
        future_hints = [h for h in hints if h.t_sec > start_t + max(user_min_sec * 0.5, 5.0)]
        end_sent, end_mode = self._pick_topic_end(
            sentences=sentences,
            start_t=start_t,
            user_min_sec=user_min_sec,
            user_max_sec=user_max_sec,
            hints=future_hints,
            silence_ranges=silence_ranges,
            video_duration_sec=video_duration_sec,
        )
        if end_sent is None:
            # Degrade (still require terminal punct). Widening the cap is the
            # last attempt to find a sentence-ending period/exclamation/question
            # inside a broader window. If THIS still returns None, the caller
            # gets a "no_terminal_end" warning + degraded quality tier — we do
            # NOT fall back to require_terminal=False because mid-sentence cuts
            # are the exact audible failure this contract prevents.
            cap = min(raw_t_end, start_t + user_max_sec * SOFT_MAX_TOLERANCE)
            end_sent = latest_sentence_ending_before(sentences, cap, require_terminal=True)
            end_mode = "fallback"
        if end_sent is None:
            warnings.append("no_terminal_end_sentence_found")
            end_t = raw_t_end
        else:
            end_t = end_sent.t_end

        # Enforce hard floor — never end before start + user_min * 0.5.
        hard_floor = start_t + max(user_min_sec * 0.5, 5.0)
        if end_t < hard_floor:
            end_t = min(hard_floor, start_t + user_max_sec * SOFT_MAX_TOLERANCE)
            warnings.append("end_time_below_min_floor")

        # Boundary quality tier.
        quality = self._classify_quality(start_sent, end_sent, cues)

        return BoundaryResult(
            t_start=float(start_t),
            t_end=float(end_t),
            start_sentence=start_sent,
            end_sentence=end_sent,
            start_mode=start_mode,
            end_mode=end_mode,
            boundary_quality=quality,
            warnings=warnings,
        )

    # ---- internal helpers ------------------------------------------------- #

    def _build_switch_hints(
        self,
        cues: list[IngestTranscriptCue],
        chapters: Sequence[dict[str, Any]] | None,
        llm_topic_segments: Iterable[dict[str, Any]] | None,
        silence_ranges: Sequence[tuple[float, float]] | None,
    ) -> list[TopicSwitchHint]:
        hints: list[TopicSwitchHint] = []
        hints.extend(_chapters_to_hints(chapters))
        hints.extend(_llm_hints_to_switch_hints(llm_topic_segments))
        # Novelty spikes as a lower-confidence backup.
        hints.extend(_novelty_hints_from_cues(cues))
        # Sort & dedupe (by time, within 3s window, keep highest-confidence).
        hints.sort(key=lambda h: (h.t_sec, -h.confidence))
        deduped: list[TopicSwitchHint] = []
        for h in hints:
            if deduped and (h.t_sec - deduped[-1].t_sec) < 3.0:
                if h.confidence > deduped[-1].confidence:
                    deduped[-1] = h
                continue
            deduped.append(h)
        return deduped

    def _pick_topic_start(
        self,
        *,
        sentences_all: list[SentenceSpan],
        window_sents: list[SentenceSpan],
        raw_t_start: float,
        query: str | None,
        silence_ranges: Sequence[tuple[float, float]] | None,
    ) -> tuple[SentenceSpan | None, str]:
        if not window_sents:
            return None, "fallback"

        # LLM-picked index.
        if self._llm_pick_start is not None and query:
            try:
                idx = self._llm_pick_start(query, window_sents[:8])
            except Exception:
                logger.exception("llm_pick_start raised; falling back to heuristic")
                idx = None
            if isinstance(idx, int) and 0 <= idx < len(window_sents):
                candidate = window_sents[idx]
                # Verify: has ≥1 query-keyword match OR non-trivial novelty.
                if self._start_looks_substantive(candidate, sentences_all, raw_t_start, query):
                    return self._silence_tiebreak(candidate, window_sents, silence_ranges, idx), "llm"

        # Heuristic: first sentence with novelty distance ≥ threshold AND ≥1 query keyword.
        query_tokens = _tokenize(query or "") if query else set()
        prior_tokens: set[str] = set()
        for s in sentences_all:
            if s.t_end < (raw_t_start - PRIOR_CONTEXT_SEC):
                continue
            if s.t_start >= raw_t_start:
                break
            prior_tokens |= _tokenize(s.text)

        for i, s in enumerate(window_sents):
            sent_tokens = _tokenize(s.text)
            if query_tokens and not (sent_tokens & query_tokens):
                continue
            if prior_tokens and _jaccard_distance(prior_tokens, sent_tokens) < MIN_NOVELTY_JACCARD:
                continue
            return self._silence_tiebreak(s, window_sents, silence_ranges, i), "novelty"

        # Last-resort: first sentence in window.
        return window_sents[0], "fallback"

    def _start_looks_substantive(
        self,
        candidate: SentenceSpan,
        all_sentences: list[SentenceSpan],
        raw_t_start: float,
        query: str | None,
    ) -> bool:
        sent_tokens = _tokenize(candidate.text)
        if not sent_tokens:
            return False
        query_tokens = _tokenize(query or "") if query else set()
        if query_tokens and not (sent_tokens & query_tokens):
            return False
        # Novelty check against prior 30s.
        prior: set[str] = set()
        for s in all_sentences:
            if s.t_end < raw_t_start - PRIOR_CONTEXT_SEC:
                continue
            if s.t_start >= candidate.t_start:
                break
            prior |= _tokenize(s.text)
        if prior and _jaccard_distance(prior, sent_tokens) < (MIN_NOVELTY_JACCARD * 0.7):
            return False
        return True

    def _pick_topic_end(
        self,
        *,
        sentences: list[SentenceSpan],
        start_t: float,
        user_min_sec: float,
        user_max_sec: float,
        hints: list[TopicSwitchHint],
        silence_ranges: Sequence[tuple[float, float]] | None,
        video_duration_sec: float | None,
    ) -> tuple[SentenceSpan | None, str]:
        soft_lo = start_t + user_min_sec * SOFT_MIN_TOLERANCE
        soft_hi = start_t + user_max_sec * SOFT_MAX_TOLERANCE
        if video_duration_sec and video_duration_sec > 0:
            soft_hi = min(soft_hi, video_duration_sec - 0.2)

        # --- Natural close: is there a topic switch inside [soft_lo, soft_hi]? ---
        in_range_hints = sorted(
            [h for h in hints if soft_lo <= h.t_sec <= soft_hi],
            key=lambda h: (-h.confidence, h.t_sec),
        )
        for hint in in_range_hints:
            candidate = latest_sentence_ending_before(sentences, hint.t_sec, require_terminal=True)
            if candidate is None or candidate.t_end <= start_t:
                continue
            if candidate.t_end < soft_lo:
                continue
            candidate = self._silence_tiebreak_end(candidate, sentences, silence_ranges)
            return candidate, "natural-close"

        # --- Truncation: pick latest terminal-punct sentence at or before user_max * 1.10 ---
        trunc_cap = start_t + user_max_sec * TRUNCATION_TOLERANCE
        if video_duration_sec and video_duration_sec > 0:
            trunc_cap = min(trunc_cap, video_duration_sec - 0.2)
        candidate = latest_sentence_ending_before(sentences, trunc_cap, require_terminal=True)
        if candidate is not None and candidate.t_end > start_t:
            if candidate.t_end < soft_lo:
                # Found a terminal sentence, but it's too short. Extend: next
                # terminal sentence, even past user_max, but bounded by SOFT_MAX_TOLERANCE.
                extended = latest_sentence_ending_before(
                    sentences, soft_hi, require_terminal=True
                )
                if extended is not None and extended.t_end >= soft_lo:
                    candidate = extended
            candidate = self._silence_tiebreak_end(candidate, sentences, silence_ranges)
            return candidate, "truncation"

        # --- Final fallback: STILL require terminal punct. Widen the window
        # one more time (to soft_hi) and require a real sentence end. If no
        # terminal-punct sentence fits anywhere in [start_t, soft_hi], return
        # None so the caller tags the clip "degraded" rather than silently
        # cutting mid-sentence. This is the whole point of Phase 2.4 —
        # begin/end on punctuation is a hard contract. ---
        fallback = latest_sentence_ending_before(sentences, soft_hi, require_terminal=True)
        return fallback, "fallback"

    def _silence_tiebreak(
        self,
        chosen: SentenceSpan,
        neighbors: list[SentenceSpan],
        silence_ranges: Sequence[tuple[float, float]] | None,
        chosen_idx: int,
    ) -> SentenceSpan:
        """If a neighbor sentence's t_start is within SILENCE_TIEBREAK_WINDOW_SEC of a
        silence range end (= speaker finished inhaling / paused), prefer it."""
        if not silence_ranges:
            return chosen
        candidates = [chosen] + neighbors[max(0, chosen_idx - 1): chosen_idx + 2]
        best = chosen
        best_score = self._silence_adjacency_score(chosen, silence_ranges, at="start")
        for cand in candidates:
            if cand is chosen:
                continue
            # Don't skip forward too far.
            if abs(cand.t_start - chosen.t_start) > 4.0:
                continue
            sc = self._silence_adjacency_score(cand, silence_ranges, at="start")
            if sc > best_score:
                best = cand
                best_score = sc
        return best

    def _silence_tiebreak_end(
        self,
        chosen: SentenceSpan,
        sentences: list[SentenceSpan],
        silence_ranges: Sequence[tuple[float, float]] | None,
    ) -> SentenceSpan:
        """If the next 350ms after chosen.t_end contains silence, chosen is already good.
        Otherwise, if the immediate predecessor sentence ends inside a silence range,
        prefer that (cleaner break)."""
        if not silence_ranges:
            return chosen
        if self._silence_adjacency_score(chosen, silence_ranges, at="end") > 0:
            return chosen
        # Try the previous terminal-punct sentence.
        idx = sentences.index(chosen) if chosen in sentences else -1
        if idx <= 0:
            return chosen
        prev = sentences[idx - 1]
        if prev.terminal_punct not in {".", "!", "?", "…"}:
            return chosen
        if abs(prev.t_end - chosen.t_end) > 6.0:
            return chosen
        if self._silence_adjacency_score(prev, silence_ranges, at="end") > 0:
            return prev
        return chosen

    @staticmethod
    def _silence_adjacency_score(
        sent: SentenceSpan,
        silence_ranges: Sequence[tuple[float, float]],
        *,
        at: str,
    ) -> float:
        target = sent.t_start if at == "start" else sent.t_end
        best = 0.0
        for (s_lo, s_hi) in silence_ranges:
            # For start: silence range should END within SILENCE_TIEBREAK_WINDOW before target.
            # For end: silence range should START within SILENCE_TIEBREAK_WINDOW after target.
            if at == "start":
                if 0.0 <= (target - s_hi) <= SILENCE_TIEBREAK_WINDOW_SEC:
                    score = 1.0 - (target - s_hi) / SILENCE_TIEBREAK_WINDOW_SEC
                    if score > best:
                        best = score
            else:
                if 0.0 <= (s_lo - target) <= SILENCE_TIEBREAK_WINDOW_SEC:
                    score = 1.0 - (s_lo - target) / SILENCE_TIEBREAK_WINDOW_SEC
                    if score > best:
                        best = score
        return best

    def _classify_quality(
        self,
        start: SentenceSpan | None,
        end: SentenceSpan | None,
        cues: list[IngestTranscriptCue],
    ) -> str:
        if start is None or end is None:
            return "degraded"
        # Check word_source of cues spanning [start.cue_start_idx, end.cue_end_idx].
        sources: set[str] = set()
        lo = start.cue_start_idx
        hi = end.cue_end_idx
        for i in range(max(0, lo), min(len(cues), hi + 1)):
            sources.add(cues[i].word_source)
        if "whisper" in sources or "openai" in sources:
            return "precise"
        if "proportional" in sources:
            return "sentence"
        return "chapter"


# --------------------------------------------------------------------------- #
# Snap-only boundary helper — for LLM-direct clip picking path
# --------------------------------------------------------------------------- #
# The ClipBoundaryEngine above does picking + snapping. The LLM-direct
# picker (services/clip_llm.py) does its own picking and only needs the
# snap half. `snap_llm_boundary` is that half — no lexical novelty, no
# LLM round-trip, just: take raw (t_start, t_end) from the LLM and force
# them onto terminal-punctuation sentence boundaries (and, when silence
# ranges are available, onto inter-word pauses so the clip doesn't pop).


# Max distance between the LLM's raw t_start/t_end and the chosen sentence
# boundary. Snapping further than this means the LLM's pick landed far from
# any sentence boundary (likely malformed) — reject rather than silently
# include several seconds of unintended audio on either side.
_SNAP_MAX_SHIFT_SEC = 5.0
# Silence gap refinement window around a chosen sentence boundary. The
# audible goal: enter/leave the clip during a natural pause, not mid-word.
_SILENCE_REFINE_WINDOW_SEC = 0.4


def _find_containing_or_next_sentence(
    sentences: Sequence[SentenceSpan], t_sec: float
) -> SentenceSpan | None:
    """First sentence whose t_end >= t_sec — the one that contains t_sec
    if one does, else the next sentence after t_sec. Returns None if t_sec
    is past every sentence's end."""
    for s in sentences:
        if s.t_end >= t_sec:
            return s
    return None


def _latest_terminal_ending_at_or_before(
    sentences: Sequence[SentenceSpan], t_sec: float
) -> SentenceSpan | None:
    """Latest sentence ending on terminal punctuation whose t_end <= t_sec.
    If t_sec falls mid-sentence, considers the containing sentence only if
    its t_end <= t_sec (it won't, by definition — but included for clarity)."""
    chosen: SentenceSpan | None = None
    for s in sentences:
        if s.t_end > t_sec:
            break
        if s.terminal_punct in {".", "!", "?", "…"}:
            chosen = s
    return chosen


def _silence_midpoint_near(
    t_sec: float,
    silence_ranges: Sequence[tuple[float, float]] | None,
    *,
    window_sec: float,
) -> float | None:
    """If a silence range overlaps [t_sec - window, t_sec + window],
    return the midpoint of the nearest overlapping range. Otherwise None.
    """
    if not silence_ranges:
        return None
    lo = t_sec - window_sec
    hi = t_sec + window_sec
    best_mid: float | None = None
    best_dist = float("inf")
    for sil_start, sil_end in silence_ranges:
        if sil_end < lo or sil_start > hi:
            continue
        mid = 0.5 * (sil_start + sil_end)
        dist = abs(mid - t_sec)
        if dist < best_dist:
            best_dist = dist
            best_mid = mid
    return best_mid


@dataclass
class SnapResult:
    """Outcome of snap_llm_boundary.

    When `snapped` is True the clip is good to serve: `t_start` / `t_end`
    are on a sentence with terminal punctuation and, where data was
    available, refined to sit inside a silence gap. When `snapped` is
    False the reason is non-empty and the clip should be rejected — never
    serve a clip that ends mid-sentence, that's the one thing this helper
    exists to prevent.
    """
    snapped: bool
    t_start: float
    t_end: float
    start_sentence: SentenceSpan | None
    end_sentence: SentenceSpan | None
    reason: str  # empty when snapped=True


def snap_llm_boundary(
    *,
    raw_t_start: float,
    raw_t_end: float,
    sentences: Sequence[SentenceSpan],
    silence_ranges: Sequence[tuple[float, float]] | None = None,
    min_sec: float = 15.0,
    max_sec: float = 60.0,
    ingest_cues: Sequence[Any] | None = None,
) -> SnapResult:
    """Snap LLM-picked raw timestamps onto terminal-punct sentence boundaries
    (and silence gaps, when available). Contract:

      * start: first sentence whose `t_start >= raw_t_start - _SNAP_SEARCH_FLOOR_SEC`.
        Start sentences don't need terminal punct at their start — only to be
        the BEGINNING of a sentence (which every SentenceSpan is by
        construction).
      * end: latest terminal-punct sentence with `t_end <= raw_t_end + _SNAP_SEARCH_CEIL_SEC`.
        If none is found, the clip is REJECTED — this is what kills the old
        silent mid-sentence fallback at clip_boundary.py:498-500.
      * duration after snap must land in [min_sec, max_sec]; otherwise reject.
      * silence refinement: when silence_ranges is provided, the final
        t_start/t_end are nudged to the midpoint of a silence gap that
        overlaps ±_SILENCE_REFINE_WINDOW_SEC of the chosen boundary. No
        silence = just use the sentence boundary verbatim (safe default —
        sentence boundaries are themselves natural breaks).

    Rejects with an explanatory `reason` rather than silently degrading,
    so reels.py can log/skip the clip instead of serving something
    mid-word.
    """
    if not sentences:
        return SnapResult(False, raw_t_start, raw_t_end, None, None, "no sentences")
    if raw_t_end <= raw_t_start:
        return SnapResult(False, raw_t_start, raw_t_end, None, None, "raw end <= start")

    sent_list = list(sentences)

    # Start: find the sentence containing raw_t_start (if any) or the next
    # sentence after raw_t_start. This snaps the boundary to the start of
    # the current thought — "begin on punctuation" = begin right after the
    # period of the previous sentence, which is the chosen sentence's t_start.
    start_sent = _find_containing_or_next_sentence(sent_list, raw_t_start)
    if start_sent is None:
        return SnapResult(
            False, raw_t_start, raw_t_end, None, None,
            "no sentence at/after raw_t_start",
        )
    if abs(start_sent.t_start - raw_t_start) > _SNAP_MAX_SHIFT_SEC:
        return SnapResult(
            False, raw_t_start, raw_t_end, start_sent, None,
            f"start shift {start_sent.t_start - raw_t_start:+.1f}s exceeds ±{_SNAP_MAX_SHIFT_SEC:.0f}s",
        )

    # End: terminal-punct sentence with t_end closest to raw_t_end (either
    # direction) within _SNAP_MAX_SHIFT_SEC. Consider both the latest one
    # ending at-or-before and the earliest one ending after — LLMs sometimes
    # round t_end down slightly, so the right answer can be a hair past
    # raw_t_end. Hard requirement on terminal punctuation; if no candidate
    # qualifies, reject rather than silently cut mid-sentence.
    before = _latest_terminal_ending_at_or_before(sent_list, raw_t_end)
    after: SentenceSpan | None = None
    for s in sent_list:
        if s.t_end > raw_t_end and s.terminal_punct in {".", "!", "?", "…"}:
            after = s
            break
    candidates = [c for c in (before, after) if c is not None]
    candidates = [c for c in candidates if abs(c.t_end - raw_t_end) <= _SNAP_MAX_SHIFT_SEC]
    if not candidates:
        return SnapResult(
            False, raw_t_start, raw_t_end, start_sent, None,
            f"no terminal-punct sentence within ±{_SNAP_MAX_SHIFT_SEC:.0f}s of raw_t_end",
        )
    end_sent = min(candidates, key=lambda c: abs(c.t_end - raw_t_end))
    if end_sent.t_end <= start_sent.t_start:
        return SnapResult(
            False, raw_t_start, raw_t_end, start_sent, end_sent,
            "end sentence precedes start sentence",
        )

    t_start = start_sent.t_start
    t_end = end_sent.t_end
    duration = t_end - t_start
    if duration < min_sec or duration > max_sec:
        return SnapResult(
            False, t_start, t_end, start_sent, end_sent,
            f"snapped duration {duration:.1f}s outside [{min_sec:.0f}, {max_sec:.0f}]",
        )

    # Silence refinement — nudge the boundary to a pause midpoint if one
    # overlaps. Preserves the terminal-punct contract (we only move within
    # _SILENCE_REFINE_WINDOW_SEC of the sentence boundary, which cannot
    # cross into a neighboring sentence).
    start_silence_mid = _silence_midpoint_near(t_start, silence_ranges, window_sec=_SILENCE_REFINE_WINDOW_SEC)
    if start_silence_mid is not None:
        t_start = start_silence_mid
    end_silence_mid = _silence_midpoint_near(t_end, silence_ranges, window_sec=_SILENCE_REFINE_WINDOW_SEC)
    if end_silence_mid is not None:
        t_end = end_silence_mid

    # Word-level + filler refinement when ingest_cues available. This takes
    # the sentence-level (t_start, t_end) and narrows to the exact acoustic
    # onset/offset of the first/last substantive (non-filler) word.
    if ingest_cues:
        t_start, t_end, _s_reason, _e_reason = refine_boundaries_word_level(
            t_start, t_end, ingest_cues, silence_ranges,
        )
        # If word-level refinement accidentally pushed duration outside
        # bounds (extremely rare — only happens when filler trim removes
        # most of a short clip), fall back to sentence-level bounds.
        if (t_end - t_start) < min_sec or (t_end - t_start) > max_sec:
            t_start = start_sent.t_start
            t_end = end_sent.t_end

    return SnapResult(True, t_start, t_end, start_sent, end_sent, "")


# --------------------------------------------------------------------------- #
# Word-level boundary refinement + filler trimming
# --------------------------------------------------------------------------- #
# Everything below this line exists to cut at the individual-word level
# instead of the sentence level. Two concrete use cases:
#
#   (a) The LLM (or the heuristic picker) returns a (t_start, t_end) that
#       lands on a sentence boundary but the first word of that sentence is
#       a filler ("so", "um", "well, okay") — we want the user to hear the
#       substantive first word, not the throat-clearing.
#   (b) The sentence's t_start is a cue-level timestamp that may include a
#       small chunk of the preceding pause; we'd rather the clip begin at
#       the actual acoustic onset of the first substantive word.
#
# Filler list composed from the working sets used by Descript, OpusClip,
# CapCut, Cleanvoice, and Riverside as of 2026. Conservative — phrases
# that are sometimes substantive (e.g., "right?") are excluded.
_FILLER_TOKENS: frozenset[str] = frozenset({
    "um", "uh", "umm", "uhh", "uhm", "er", "err", "eh", "ah", "hm", "hmm", "mm",
    "okay", "ok", "alright", "so", "well", "now", "like", "basically",
    "literally", "actually", "honestly", "obviously", "anyway", "anyways",
    "right", "yeah", "yep", "yup", "nope", "nah",
})
_FILLER_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
    ("you", "know"), ("i", "mean"), ("sort", "of"), ("kind", "of"),
    ("you", "see"), ("like", "i"), ("like", "so"), ("so", "like"),
    ("okay", "so"), ("alright", "so"), ("ok", "so"), ("so", "um"),
    ("so", "yeah"), ("um", "so"), ("right", "so"), ("well", "um"),
})

# How much inter-word time separates a "real pause" from natural between-
# word ms-scale gaps. 150ms is a common empirical breakpoint from speech
# research — shorter and it's just coarticulation/breath, longer and it's
# a perceptible pause a listener notices.
_INTER_WORD_PAUSE_SEC = 0.15
# Filler trimming is bounded — we won't walk more than this many words in
# from either edge looking for the first non-filler. Prevents pathological
# cases (all-filler clip, or a transcript pun about "um") from chewing the
# whole clip.
_MAX_FILLER_TRIM_WORDS = 4


def _cue_has_word_timings(cue: Any) -> bool:
    """Duck-type: cue has a non-empty `words` list with .start/.end."""
    words = getattr(cue, "words", None)
    if not words:
        return False
    first = words[0]
    return hasattr(first, "start") and hasattr(first, "end") and hasattr(first, "text")


def _collect_word_timings_in_range(
    cues: Sequence[Any],
    t_lo: float,
    t_hi: float,
) -> list[tuple[str, float, float]]:
    """Flatten cue.words into a list of (text, t_start, t_end) triples whose
    timestamps land inside [t_lo - 0.3s, t_hi + 2.0s]. The upper slop is
    generous because the end-boundary refiner needs the next-word-after-
    end to compute the silence-gap midpoint, and a gap can be >1s on a
    clean speaker pause. Returns [] when no cue has word-level timing.
    """
    out: list[tuple[str, float, float]] = []
    for cue in cues:
        if not _cue_has_word_timings(cue):
            continue
        for w in cue.words:
            try:
                ws = float(w.start)
                we = float(w.end)
            except (TypeError, ValueError):
                continue
            if we < t_lo - 0.3 or ws > t_hi + 2.0:
                continue
            txt = str(w.text or "").strip()
            if not txt:
                continue
            out.append((txt, ws, we))
    out.sort(key=lambda tup: tup[1])
    return out


def _word_is_filler(word_text: str) -> bool:
    """True when a single-word token is in the filler set. Strips trailing
    punctuation since Whisper sometimes attaches a comma to filler words."""
    w = word_text.lower().strip(",.?!;:\"'()[]{}…")
    return w in _FILLER_TOKENS


def _bigram_is_filler(a: str, b: str) -> bool:
    aa = a.lower().strip(",.?!;:\"'()[]{}…")
    bb = b.lower().strip(",.?!;:\"'()[]{}…")
    return (aa, bb) in _FILLER_BIGRAMS


def _refine_start_to_word_boundary(
    words_in_range: list[tuple[str, float, float]],
    sentence_t_start: float,
    silence_ranges: Sequence[tuple[float, float]] | None,
) -> tuple[float, str]:
    """Refine a sentence-level start timestamp to word-level precision.
    Strategy:

        1. Find the first word at or after `sentence_t_start` — this is
           the actual acoustic onset of the first word of the sentence,
           which can differ from `sentence_t_start` by up to a few
           hundred ms when the cue boundary includes a pause.
        2. If an inter-word silence (>= _INTER_WORD_PAUSE_SEC) exists
           right before the chosen word, snap to the silence midpoint —
           cleaner listener experience than cutting exactly on the first
           phoneme.

    Filler trimming across the sentence boundary is intentionally NOT
    done: it would shift the clip start off the terminal-punct contract
    ("the word before the clip must end with .!?…"). Fillers inside the
    sentence still get scored against by `_score_window`'s filler-density
    penalty, so filler-heavy windows lose at selection time.
    """
    if not words_in_range:
        return sentence_t_start, "no-word-timing"
    chosen_idx: int | None = None
    for i, (_, ws, _we) in enumerate(words_in_range):
        if ws >= sentence_t_start - 0.1:
            chosen_idx = i
            break
    if chosen_idx is None:
        return sentence_t_start, "no-word-at-start"
    chosen = words_in_range[chosen_idx]
    refined_t_start = chosen[1]
    if chosen_idx > 0:
        prev_end = words_in_range[chosen_idx - 1][2]
        gap = chosen[1] - prev_end
        if gap >= _INTER_WORD_PAUSE_SEC:
            refined_t_start = prev_end + 0.5 * gap
    silence_mid = _silence_midpoint_near(
        chosen[1], silence_ranges, window_sec=_SILENCE_REFINE_WINDOW_SEC,
    )
    if silence_mid is not None:
        refined_t_start = silence_mid
    return refined_t_start, "word-snap"


def _refine_end_to_word_boundary(
    words_in_range: list[tuple[str, float, float]],
    sentence_t_end: float,
    silence_ranges: Sequence[tuple[float, float]] | None,
) -> tuple[float, str]:
    """Mirror of `_refine_start_to_word_boundary` for the end boundary.
    Finds the last word that ends at or before `sentence_t_end` and snaps
    to a trailing inter-word silence midpoint when one exists.

    As with the start, filler trimming is NOT applied — that would move
    the end off the terminal-punct contract.
    """
    if not words_in_range:
        return sentence_t_end, "no-word-timing"
    chosen_idx: int | None = None
    for i in range(len(words_in_range) - 1, -1, -1):
        if words_in_range[i][2] <= sentence_t_end + 0.1:
            chosen_idx = i
            break
    if chosen_idx is None:
        return sentence_t_end, "no-word-at-end"
    chosen = words_in_range[chosen_idx]
    refined_t_end = chosen[2]
    if chosen_idx + 1 < len(words_in_range):
        next_start = words_in_range[chosen_idx + 1][1]
        gap = next_start - chosen[2]
        if gap >= _INTER_WORD_PAUSE_SEC:
            refined_t_end = chosen[2] + 0.5 * gap
    silence_mid = _silence_midpoint_near(
        chosen[2], silence_ranges, window_sec=_SILENCE_REFINE_WINDOW_SEC,
    )
    if silence_mid is not None:
        refined_t_end = silence_mid
    return refined_t_end, "word-snap"


def refine_boundaries_word_level(
    t_start: float,
    t_end: float,
    ingest_cues: Sequence[Any] | None,
    silence_ranges: Sequence[tuple[float, float]] | None,
) -> tuple[float, float, str, str]:
    """Adjust sentence-level (t_start, t_end) to word-level boundaries,
    trimming filler words and snapping to inter-word silence midpoints.

    Returns ``(refined_t_start, refined_t_end, start_reason, end_reason)``.
    When no word-level data is available the input timestamps pass through
    unchanged with reason ``"no-word-timing"``. Never returns boundaries
    that would invert the clip (``refined_t_end > refined_t_start``
    guaranteed); when filler trimming would cross, returns the originals.
    """
    if not ingest_cues or t_end <= t_start:
        return t_start, t_end, "no-cues", "no-cues"
    words_in = _collect_word_timings_in_range(ingest_cues, t_start, t_end)
    if not words_in:
        return t_start, t_end, "no-word-timing", "no-word-timing"
    new_start, start_reason = _refine_start_to_word_boundary(words_in, t_start, silence_ranges)
    new_end, end_reason = _refine_end_to_word_boundary(words_in, t_end, silence_ranges)
    if new_end <= new_start:
        return t_start, t_end, "trim-inverted", "trim-inverted"
    return new_start, new_end, start_reason, end_reason


# --------------------------------------------------------------------------- #
# Strong heuristic clip picker — "as robust as an AI"
# --------------------------------------------------------------------------- #
# Replaces the old Jaccard-novelty + lexical-keyword picker. Scoring
# combines:
#   1. Semantic similarity to the query via sentence-transformer embeddings
#      (when available on Railway) — the single strongest signal.
#   2. TF-IDF-weighted keyword coverage so content terms dominate over
#      function words.
#   3. Filler-density penalty (see _FILLER_TOKENS) so um/uh-heavy passages
#      lose to clean ones.
#   4. Silence-start bonus when the window opens right after a >1s pause
#      (strong cue that a new thought is starting).
#   5. Intro/outro dampening (first 10s and last 15s of the video almost
#      never contain the "most important" passage).
# The window is enumerated at sentence granularity: every terminal-punct-
# to-terminal-punct run whose duration falls inside [user_min, user_max].
# This cheaply guarantees the chosen window starts and ends on punctuation
# before the word-level refiner runs — matching the contract that no clip
# ever ends mid-sentence.


_CONTENT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")
# TF-IDF over sliding windows — used when embeddings aren't available. Each
# "document" is a window, IDF lowers the weight of words appearing in many
# windows (common function words, topic preamble, etc).
_HEURISTIC_INTRO_PENALTY_SEC = 10.0
_HEURISTIC_OUTRO_PENALTY_SEC = 15.0
_HEURISTIC_SILENCE_BONUS_WINDOW_SEC = 1.0
_HEURISTIC_SILENCE_BONUS_MIN_GAP_SEC = 1.0


def _extract_content_words(text: str) -> list[str]:
    toks = [m.group(0).lower() for m in _CONTENT_WORD_RE.finditer(text or "")]
    return [t for t in toks if t not in _FILLER_TOKENS and t not in _STOPWORDS and len(t) > 1]


def _cosine_similarity(a: "Any", b: "Any") -> float:
    """Cosine similarity between two 1-D numpy vectors. Imports numpy
    locally to keep clip_boundary importable on machines without it — the
    fallback scorer still works."""
    try:
        import numpy as np
    except ImportError:
        return 0.0
    a_arr = np.asarray(a, dtype="float32")
    b_arr = np.asarray(b, dtype="float32")
    an = float((a_arr * a_arr).sum()) ** 0.5
    bn = float((b_arr * b_arr).sum()) ** 0.5
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float((a_arr * b_arr).sum()) / (an * bn)


def _enumerate_candidate_windows(
    sentences: Sequence[SentenceSpan],
    min_sec: float,
    max_sec: float,
) -> list[tuple[int, int]]:
    """Return every (start_idx, end_idx) pair over the sentence list where:
      * the END sentence has terminal punctuation (so the window ends on
        a complete thought);
      * total duration falls inside [min_sec, max_sec].
    Each pair indexes the sentences slice [start_idx : end_idx + 1]."""
    windows: list[tuple[int, int]] = []
    n = len(sentences)
    for i in range(n):
        for j in range(i, n):
            s_j = sentences[j]
            if s_j.terminal_punct not in {".", "!", "?", "…"}:
                continue
            dur = s_j.t_end - sentences[i].t_start
            if dur < min_sec:
                continue
            if dur > max_sec:
                break  # durations only grow as j increases
            windows.append((i, j))
    return windows


def _enumerate_cue_fallback_windows(
    cues: Sequence[Any],
    min_sec: float,
    max_sec: float,
) -> list[tuple[int, int, float, float]]:
    """Fallback enumerator for transcripts without terminal punctuation
    (YouTube auto-captions, comma-spliced text, run-on transcripts). Uses
    cue boundaries as synthetic "soft-sentence" markers: each (t_start,
    t_end) pair is a span of cues whose duration lands in [min, max].

    Returns ``(i, j, t_start, t_end)`` tuples — i/j index the cue list;
    t_start/t_end are the actual timestamps the clip will use. Unlike
    `_enumerate_candidate_windows`, this does NOT guarantee terminal
    punctuation — it guarantees only that the boundary aligns with a cue
    edge (i.e., the speaker's utterance gap), which is the closest thing
    to a "sentence end" we have when pysbd can't find one.
    """
    windows: list[tuple[int, int, float, float]] = []
    n = len(cues)
    if n == 0:
        return windows
    for i in range(n):
        try:
            t_start = float(cues[i].start)
        except (AttributeError, TypeError, ValueError):
            continue
        for j in range(i, n):
            try:
                t_end = float(cues[j].end)
            except (AttributeError, TypeError, ValueError):
                continue
            dur = t_end - t_start
            if dur < min_sec:
                continue
            if dur > max_sec:
                break
            windows.append((i, j, t_start, t_end))
    return windows


def _score_window(
    *,
    sentences_slice: list[SentenceSpan],
    query_words: list[str],
    global_df: dict[str, int],
    num_windows: int,
    query_embedding: "Any",
    embed_func: Any | None,
    video_duration_sec: float | None,
    silence_ranges: Sequence[tuple[float, float]] | None,
    user_target_sec: float,
) -> float:
    """Score one candidate window. Returns a float in roughly [-1, 2]."""
    text = " ".join(s.text for s in sentences_slice)
    window_words = _extract_content_words(text)
    word_count = max(1, len(window_words))

    # --- 1. Semantic similarity (strongest signal when available). ---
    semantic = 0.0
    if embed_func is not None and query_embedding is not None:
        try:
            window_emb = embed_func([text])
            if window_emb is not None and len(window_emb) > 0:
                semantic = _cosine_similarity(query_embedding, window_emb[0])
        except Exception:
            semantic = 0.0

    # --- 2. TF-IDF weighted keyword coverage. ---
    tf_idf = 0.0
    if query_words:
        # Term frequency within window (normalized).
        from collections import Counter
        tf = Counter(window_words)
        denom = sum(tf.values()) or 1
        query_terms = set(query_words)
        for qt in query_terms:
            term_tf = tf.get(qt, 0) / denom
            if term_tf == 0.0:
                continue
            # IDF: log((N+1)/(df+1)) + 1 (smoothed).
            df = global_df.get(qt, 0)
            idf = 1.0
            if num_windows > 0:
                import math
                idf = math.log((num_windows + 1) / (df + 1)) + 1.0
            tf_idf += term_tf * idf
        # Normalize so tf_idf is roughly [0, 1] for typical queries.
        tf_idf = min(1.0, tf_idf)

    # --- 3. Filler density penalty. ---
    filler_count = sum(
        1 for m in _CONTENT_WORD_RE.finditer(text)
        if m.group(0).lower() in _FILLER_TOKENS
    )
    filler_ratio = filler_count / word_count
    filler_penalty = min(0.35, filler_ratio * 0.6)

    # --- 4. Silence-start bonus. ---
    t_start = sentences_slice[0].t_start
    silence_bonus = 0.0
    if silence_ranges:
        for sil_start, sil_end in silence_ranges:
            gap = sil_end - sil_start
            if (
                gap >= _HEURISTIC_SILENCE_BONUS_MIN_GAP_SEC
                and abs(sil_end - t_start) <= _HEURISTIC_SILENCE_BONUS_WINDOW_SEC
            ):
                silence_bonus = 0.12
                break

    # --- 5. Intro/outro dampening. ---
    intro_outro_penalty = 0.0
    if t_start < _HEURISTIC_INTRO_PENALTY_SEC:
        intro_outro_penalty = 0.18
    elif (
        video_duration_sec is not None
        and (video_duration_sec - sentences_slice[-1].t_end) < _HEURISTIC_OUTRO_PENALTY_SEC
    ):
        intro_outro_penalty = 0.12

    # --- 6. Prefer windows near the user's target duration. ---
    duration = sentences_slice[-1].t_end - t_start
    # Penalty scales with how far duration strays from user_target_sec,
    # normalized by user_target_sec so the scale is comparable to other terms.
    target_delta = abs(duration - user_target_sec) / max(1.0, user_target_sec)
    target_penalty = min(0.2, target_delta * 0.25)

    # Weighted sum — semantic dominates when present, lexical carries more
    # weight when semantic is unavailable (emb_func=None).
    if embed_func is not None:
        score = 0.55 * semantic + 0.25 * tf_idf
    else:
        score = 0.6 * tf_idf  # no embeddings → tf-idf is the primary signal
    score += silence_bonus
    score -= filler_penalty
    score -= intro_outro_penalty
    score -= target_penalty
    return score


@dataclass
class HeuristicPickResult:
    t_start: float
    t_end: float
    start_sentence: SentenceSpan
    end_sentence: SentenceSpan
    score: float
    signal_summary: str  # e.g. "semantic=0.72 tf-idf=0.15 fillers=0.04"


def pick_clip_heuristic(
    *,
    query: str,
    cues: Sequence[Any] | None,  # list[IngestTranscriptCue] when available
    sentences: Sequence[SentenceSpan],
    silence_ranges: Sequence[tuple[float, float]] | None,
    user_min_sec: float,
    user_max_sec: float,
    user_target_sec: float,
    video_duration_sec: float | None,
    embed_func: Any | None = None,
) -> HeuristicPickResult | None:
    """Picker that runs without any LLM call. Scores every
    terminal-punct-to-terminal-punct window that fits [user_min, user_max]
    on semantic similarity + TF-IDF + filler/silence/intro signals, then
    returns the top-scoring window. Returns None when no qualifying window
    exists (e.g., transcript too short, no terminal-punct sentences).

    ``embed_func`` is an optional ``Callable[[list[str]], np.ndarray]`` —
    pass ``EmbeddingService().embed_local`` to activate semantic scoring.
    Omit to fall back to lexical-only (still stronger than the old Jaccard
    picker because it's TF-IDF-weighted).
    """
    sent_list = [s for s in sentences if s is not None]
    windows = _enumerate_candidate_windows(sent_list, user_min_sec, user_max_sec)
    using_cue_fallback = False
    cue_windows: list[tuple[int, int, float, float]] = []
    if not windows:
        # No sentence windows ended on terminal punctuation — common for
        # YouTube auto-captions and comma-spliced transcripts. Fall back
        # to cue-boundary windows so we still produce SOMETHING. The
        # word-level refiner at the tail of this function still trims
        # leading/trailing fillers for acoustically clean edges.
        if cues:
            cue_windows = _enumerate_cue_fallback_windows(cues, user_min_sec, user_max_sec)
        if not cue_windows:
            return None
        using_cue_fallback = True

    query_words = _extract_content_words(query)
    # Precompute a query embedding once, not once per window.
    query_embedding = None
    if embed_func is not None and query:
        try:
            qemb = embed_func([query])
            if qemb is not None and len(qemb) > 0:
                query_embedding = qemb[0]
        except Exception:
            query_embedding = None

    # Build candidate-window metadata. Sentence-window path uses
    # sent_list[i:j+1] text; cue-fallback path uses cues[i:j+1] text.
    # Both feed the same scoring function — only the "slice" representation
    # changes, plus the chosen boundary sentences for SnapResult.
    def _window_text(widx: int) -> str:
        if using_cue_fallback:
            i, j, _, _ = cue_windows[widx]
            return " ".join(str(getattr(cues[k], "text", "") or "") for k in range(i, j + 1))
        i, j = windows[widx]
        return " ".join(s.text for s in sent_list[i : j + 1])

    def _window_bounds(widx: int) -> tuple[float, float]:
        if using_cue_fallback:
            _, _, t0, t1 = cue_windows[widx]
            return t0, t1
        i, j = windows[widx]
        return sent_list[i].t_start, sent_list[j].t_end

    num_windows = len(cue_windows) if using_cue_fallback else len(windows)

    # Build global document frequency for TF-IDF weighting. "Documents"
    # are the candidate windows themselves — gives us topic-specific IDF
    # without needing a corpus.
    global_df: dict[str, int] = {}
    for widx in range(num_windows):
        words = set(_extract_content_words(_window_text(widx)))
        for w in words:
            global_df[w] = global_df.get(w, 0) + 1

    best: HeuristicPickResult | None = None
    best_bounds: tuple[float, float] | None = None
    best_idx: int = -1
    for widx in range(num_windows):
        text = _window_text(widx)
        t_start, t_end = _window_bounds(widx)
        # Build a synthetic single-sentence slice for the cue-fallback
        # path so _score_window's sentence-typed interface stays uniform.
        if using_cue_fallback:
            synthetic = SentenceSpan(
                text=text,
                t_start=t_start,
                t_end=t_end,
                cue_start_idx=cue_windows[widx][0],
                cue_end_idx=cue_windows[widx][1],
                word_start_idx=0,
                word_end_idx=0,
                terminal_punct="",  # cue-fallback path has no terminal punct
                confidence=0.4,
            )
            sentences_slice = [synthetic]
        else:
            i, j = windows[widx]
            sentences_slice = list(sent_list[i : j + 1])
        score = _score_window(
            sentences_slice=sentences_slice,
            query_words=query_words,
            global_df=global_df,
            num_windows=num_windows,
            query_embedding=query_embedding,
            embed_func=embed_func,
            video_duration_sec=video_duration_sec,
            silence_ranges=silence_ranges,
            user_target_sec=user_target_sec,
        )
        # Cue-fallback windows pay a small quality penalty — real sentence
        # windows are objectively better when both are candidates.
        if using_cue_fallback:
            score -= 0.05
        if best is None or score > best.score:
            summary = (
                f"semantic={'y' if embed_func else 'n'} "
                f"mode={'cue-fallback' if using_cue_fallback else 'sentence'} "
                f"tfidf_terms={len(query_words)} duration={t_end - t_start:.1f}s"
            )
            best = HeuristicPickResult(
                t_start=t_start,
                t_end=t_end,
                start_sentence=sentences_slice[0],
                end_sentence=sentences_slice[-1],
                score=score,
                signal_summary=summary,
            )
            best_bounds = (t_start, t_end)
            best_idx = widx

    if best is None:
        return None

    # Apply word-level + filler trimming to the chosen sentence-level
    # window. Only narrows t_start / t_end; the underlying sentence choice
    # (which guarantees terminal punctuation) is preserved.
    refined_start, refined_end, _, _ = refine_boundaries_word_level(
        best.t_start, best.t_end, cues, silence_ranges,
    )
    best.t_start = refined_start
    best.t_end = refined_end
    return best


__all__ = [
    "BOUNDARY_ENGINE_VERSION",
    "ClipBoundaryEngine",
    "BoundaryResult",
    "TopicSwitchHint",
    "SnapResult",
    "snap_llm_boundary",
    "refine_boundaries_word_level",
    "HeuristicPickResult",
    "pick_clip_heuristic",
]
