"""ReelAI's one-pass Gemini educational clip selector.

A single Gemini pass reads the timestamped supadata transcript and returns every
substantive teaching topic as a clip {title, start, end}. No punctuation restoration,
no structure understanding, no local-Whisper refine, no multimodal — the whole
``understand → assemble → refine`` chain is replaced by one comprehension call.

Boundaries: the model picks a start/end transcript LINE (a real caption-chunk with a
real timestamp) plus a short opening/closing QUOTE. Quote matching supplies the
semantic boundary; nearby caption gaps provide a cheap, audio-free pause boundary.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .. import config
from ..llm import llm_json

ProgressCb = Optional[Callable[[float, str], None]]

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


# ── LLM schema ───────────────────────────────────────────────────────────────
class _Topic(BaseModel):
    title: str
    start_line: int
    end_line: int
    start_quote: str
    end_quote: str
    reason: str = ""
    facet: str = "other"
    kind: Optional[str] = None
    informativeness: Optional[float] = None
    topic_relevance: Optional[float] = None
    self_contained: Optional[bool] = None
    difficulty: float = 0.5   # 0 = assumes no prior knowledge, 1 = expert-level
    summary: str = ""
    takeaways: list[str] = Field(default_factory=list)
    match_reason: str = ""
    # Keep this deliberately loose. A malformed assessment must be discarded
    # without making Pydantic reject the otherwise usable clip proposal.
    assessment: dict | None = None


# Only explicitly educational kinds ship; missing and novel labels fail closed.
_ACCEPT_KINDS = {"content", "educational"}


def _norm_informativeness(value: float) -> float:
    """Clamp to [0, 1], tolerating models that answer on a 0-10 or 0-100 scale
    (7 → 0.7, 85 → 0.85) instead of saturating every mis-scaled score to 1.0."""
    v = float(value)
    if v > 10.0:
        v = v / 100.0
    elif v > 1.0:
        v = v / 10.0
    return max(0.0, min(1.0, v))


def _norm_optional_confidence(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return _norm_informativeness(value)


class _Plan(BaseModel):
    topics: list[_Topic]


def _mmss(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def _prompts(lines: str, n: int, topic: str = "") -> "tuple[str, str]":
    topic_rule = ""
    if topic.strip():
        topic_rule = (
            f"The viewer is studying: {topic.strip()!r}. Only return clips that TEACH "
            "material relevant to that topic; skip unrelated sections entirely.\n"
        )
    system = (
        "You select self-contained CLIPS from a lecture/talk transcript for a short-form "
        "learning feed. First read and understand the WHOLE transcript. Then pick the "
        "SUBSTANTIVE teaching moments — one coherent idea, concept, worked example, or "
        "section, taught from its introduction through to its natural conclusion. Skip pure "
        "filler (greetings, admin, 'like and subscribe', tangents), course-logistics intros, "
        "and wrap-up outros.\n" + topic_rule +
        "For every clip return: title; start_line (the line where the idea is INTRODUCED); "
        "end_line (the line where it CLOSES); start_quote (the first ~6 words spoken at the "
        "start, copied verbatim from that line); end_quote (the last ~6 words, verbatim); a "
        "short reason; kind — one of content|educational|intro|outro|admin|promo; "
        "informativeness — 0.0 to 1.0, how much a motivated student learns from this clip "
        "ALONE; topic_relevance — 0.0 to 1.0, how directly it teaches the viewer's topic; "
        "self_contained — true only when it makes sense without omitted context; "
        "difficulty — 0.0 to 1.0, the prior knowledge the clip ASSUMES (0.1: no background, "
        "first exposure; 0.5: comfortable with the basics; 0.9: graduate/expert material); "
        "summary — a grounded 1-2 sentence explanation of what this exact clip teaches; "
        "takeaways — 2-4 concise, non-overlapping facts or ideas from this exact clip; "
        "match_reason — one topic-specific sentence explaining why this clip helps this viewer "
        "(name the idea taught; never write a generic recommendation); assessment — exactly one "
        "object with prompt, options (exactly four distinct strings), correct_index (0-3), and a "
        "grounded explanation that states why the correct option follows from the clip. Ask about "
        "the central idea, make exactly one option correct, and do not use 'all of the above'. "
        "Rules: (1) a clip must START at the beginning of the idea and END at its end — "
        "never mid-thought; (2) contextual overlap is allowed when two complete ideas share "
        "setup; (3) go in chronological order; (4) prefer complete clips of 20-90 seconds. "
        "Split longer sections into complete subtopics. A complete clip may be up to 180 "
        "seconds, but never force or truncate a longer section; (5) line indices range from 0 to "
        f"{n - 1} — never exceed {n - 1}."
    )
    user = (
        f"Transcript ({n} lines, each formatted `[index] MM:SS text`):\n\n" + lines +
        "\n\nReturn every substantive teaching clip as {title, start_line, end_line, "
        "start_quote, end_quote, reason, facet, kind, informativeness, topic_relevance, "
        "self_contained, difficulty, summary, takeaways, match_reason, assessment: "
        "{prompt, options, correct_index, explanation}}."
    )
    return system, user


# ── fine-snap (interpolated word times) ──────────────────────────────────────
def _toks(s: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(s or "")]


def _locate_quote(words: list[dict], quote: str, lo_t: float, hi_t: float,
                  want: str) -> Optional[float]:
    """Find ``quote`` among ``words`` whose times fall in [lo_t, hi_t] and return the
    boundary time: the matched window's first-word start (want="start") or last-word end
    (want="end"). Uses rapidfuzz over a sliding token window; None if no good match."""
    q = _toks(quote)
    if not q or not words:
        return None
    try:
        from rapidfuzz import fuzz
    except Exception:
        return None
    qn = min(len(q), 6)
    target = " ".join(q[:qn] if want == "start" else q[-qn:])
    # one aligned list of (token, start, end) for words inside the time window
    toks: list[tuple[str, float, float]] = []
    for w in words:
        t0 = float(w.get("start", 0.0))
        if not (lo_t - 1e-6 <= t0 <= hi_t + 1e-6):
            continue
        wt = _toks(w.get("word", ""))
        if wt:
            toks.append((wt[0], t0, float(w.get("end", t0))))
    if len(toks) < qn:
        return None
    best_score, best_t = 0.0, None
    for i in range(len(toks) - qn + 1):
        window = " ".join(tok for tok, _, _ in toks[i:i + qn])
        score = fuzz.ratio(target, window)
        if score > best_score:
            best_score = score
            best_t = toks[i][1] if want == "start" else toks[i + qn - 1][2]
    return best_t if best_score >= 70 else None


def _caption_gap_boundary(
    segs: list[dict], semantic_time: float, *, direction: str,
    min_gap_s: float = 0.25, max_move_s: float = 2.0,
) -> float:
    """Move outward to the nearest qualifying caption-gap midpoint."""
    candidates: list[float] = []
    for left, right in zip(segs, segs[1:]):
        gap_start = float(left.get("end", 0.0))
        gap_end = float(right.get("start", gap_start))
        if gap_end - gap_start < min_gap_s:
            continue
        midpoint = (gap_start + gap_end) / 2.0
        delta = semantic_time - midpoint if direction == "start" else midpoint - semantic_time
        if -1e-9 <= delta <= max_move_s + 1e-9:
            candidates.append(midpoint)
    if not candidates:
        return semantic_time
    if direction == "start":
        return max(candidates)
    return min(candidates)


def _near_duplicate(a: dict, b: dict, threshold: float = 0.8) -> bool:
    overlap = min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"]))
    if overlap <= 0:
        return False
    shorter = min(float(a["end"]) - float(a["start"]), float(b["end"]) - float(b["start"]))
    return shorter > 0 and overlap / shorter >= threshold


def _clean_list(value: object, *, minimum: int, maximum: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = " ".join(str(item or "").split()).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= maximum:
            break
    return cleaned if len(cleaned) >= minimum else []


def _validated_assessment(value: object, *, grounding_text: str) -> dict | None:
    """Return a privacy-ready stored question or None.

    Validation happens after the clip proposal is accepted so a bad question
    never rejects useful educational content.
    """
    if not isinstance(value, dict):
        return None
    prompt = " ".join(str(value.get("prompt") or "").split()).strip()
    explanation = " ".join(str(value.get("explanation") or "").split()).strip()
    raw_options = value.get("options")
    if not isinstance(raw_options, list) or len(raw_options) != 4:
        return None
    options = _clean_list(raw_options, minimum=4, maximum=4)
    raw_correct_index = value.get("correct_index")
    if isinstance(raw_correct_index, bool) or not isinstance(raw_correct_index, int):
        return None
    correct_index = raw_correct_index
    if not prompt or not explanation or len(options) != 4 or not 0 <= correct_index < 4:
        return None

    # Require a concrete lexical anchor to the clip or its generated details.
    # This is intentionally modest: it blocks generic praise/rationales without
    # rejecting concise explanations that use ordinary connective words.
    source_tokens = {t for t in _toks(grounding_text) if len(t) >= 4}
    explanation_tokens = {t for t in _toks(explanation) if len(t) >= 4}
    if source_tokens and not (source_tokens & explanation_tokens):
        return None
    return {
        "prompt": prompt,
        "options": options,
        "correct_index": correct_index,
        "explanation": explanation,
    }


# ── plan → clips ─────────────────────────────────────────────────────────────
def _plan_to_clips(plan: _Plan, segs: list[dict], words: list[dict],
                   settings: dict) -> list[dict]:
    n = len(segs)
    fine = settings.get("segment_fine_snap")
    fine = config.SEGMENT_FINE_SNAP if fine is None else bool(fine)
    min_s = 1.0
    max_s = 180.0
    info_min = settings.get("segment_informativeness_min")
    info_min = max(0.6, config.SEGMENT_INFORMATIVENESS_MIN if info_min is None else float(info_min))
    relevance_min = settings.get("segment_topic_relevance_min")
    relevance_min = max(0.6, config.SEGMENT_TOPIC_RELEVANCE_MIN if relevance_min is None else float(relevance_min))

    raw: list[dict] = []
    for tp in plan.topics:
        kind = (tp.kind or "").strip().lower()
        info = _norm_optional_confidence(tp.informativeness)
        relevance = _norm_optional_confidence(tp.topic_relevance)
        contained = tp.self_contained
        if kind not in _ACCEPT_KINDS:
            continue
        if not tp.start_quote.strip() or not tp.end_quote.strip():
            continue
        if info is None or relevance is None or contained is not True:
            continue
        if info < info_min or relevance < relevance_min:
            continue
        a = max(0, min(int(tp.start_line), n - 1))
        b = max(a, min(int(tp.end_line), n - 1))
        start = float(segs[a]["start"])
        end = float(segs[b]["end"])
        if fine and words:
            # snap start within [chunk a start-2s, chunk a end]; end within [chunk b start, chunk b end+2s]
            st = _locate_quote(words, tp.start_quote, start - 2.0, float(segs[a]["end"]), "start")
            en = _locate_quote(words, tp.end_quote, float(segs[b]["start"]), end + 2.0, "end")
            if st is not None and st < end:
                start = st
            if en is not None and en > start:
                end = en
        start = _caption_gap_boundary(segs, start, direction="start")
        end = _caption_gap_boundary(segs, end, direction="end")
        clip_text = " ".join(
            str(seg.get("text") or "").strip() for seg in segs[a:b + 1]
        )
        summary = " ".join((tp.summary or "").split()).strip()
        takeaways = _clean_list(tp.takeaways, minimum=2, maximum=4)
        match_reason = " ".join((tp.match_reason or "").split()).strip()
        assessment = _validated_assessment(
            tp.assessment,
            grounding_text=" ".join(
                part for part in (clip_text, tp.title, summary, " ".join(takeaways)) if part
            ),
        )
        raw.append({"start": round(start, 3), "end": round(end, 3),
                    "title": (tp.title or "").strip(),
                    "facet": (tp.facet or "other").strip() or "other",
                    "reason": (tp.reason or "").strip(),
                    "kind": kind,
                    "informativeness": info,
                    "topic_relevance": relevance,
                    "self_contained": True,
                    "difficulty": _norm_informativeness(tp.difficulty),
                    "summary": summary,
                    "takeaways": takeaways,
                    "match_reason": match_reason,
                    "assessment": assessment})

    eligible = [c for c in raw if min_s <= c["end"] - c["start"] <= max_s]
    # Highest-quality member of a near-duplicate cluster wins; contextual
    # overlap below the threshold remains untouched.
    quality_order = sorted(
        eligible,
        key=lambda c: (
            c["informativeness"] + c["topic_relevance"],
            -(c["end"] - c["start"]),
        ),
        reverse=True,
    )
    clips: list[dict] = []
    for candidate in quality_order:
        if any(_near_duplicate(candidate, kept) for kept in clips):
            continue
        clips.append(candidate)
    clips.sort(key=lambda c: (c["start"], c["end"]))

    max_clips = min(40, int(settings.get("max_clips") or config.SEGMENT_MAX_CLIPS))
    clips = clips[:max_clips]
    for i, c in enumerate(clips):
        c["sequence_index"] = i + 1
    return clips


# ── public entry point ───────────────────────────────────────────────────────
def segment_clips(transcript: dict, settings: dict,
                  progress: ProgressCb = None, topic: str = "") -> "tuple[list[dict], str]":
    """One Gemini comprehension pass → curated teaching clips (topic-relevant when
    ``topic`` is given; intro/outro/low-informativeness/over-length segments dropped).
    Returns (clips_spec, notes)."""
    segs = transcript.get("segments") or []
    words = transcript.get("words") or []
    if not segs:
        return [], "No transcript segments to segment."
    n = len(segs)
    lines = "\n".join(f"[{i}] {_mmss(s.get('start', 0.0))} {(s.get('text') or '').strip()}"
                      for i, s in enumerate(segs))
    system, user = _prompts(lines, n, topic)
    model = settings.get("segment_model") or config.SEGMENT_MODEL

    if progress:
        progress(0.1, "Understanding the transcript…")
    plan = llm_json(system, user, _Plan, temperature=0.2, model=model,
                    max_output_tokens=config.SEGMENT_MAX_OUTPUT_TOKENS,
                    should_cancel=settings.get("should_cancel"))
    if progress:
        progress(0.85, "Placing clip boundaries…")
    clips = _plan_to_clips(plan, segs, words, settings)
    if progress:
        progress(1.0, f"{len(clips)} clip(s) ready")
    notes = f"{len(clips)} topic clip(s) from {n} transcript segments (Gemini-segment engine)."
    return clips, notes
