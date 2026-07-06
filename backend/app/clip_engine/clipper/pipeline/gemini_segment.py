"""Gemini-segment clip engine (opt-in, ``clip_engine="gemini"``).

A single Gemini pass reads the timestamped supadata transcript and returns every
substantive teaching topic as a clip {title, start, end}. No punctuation restoration,
no structure understanding, no local-Whisper refine, no multimodal — the whole
``understand → assemble → refine`` chain is replaced by one comprehension call.

Boundaries: the model picks a start/end transcript LINE (a real caption-chunk with a
real timestamp) plus a short opening/closing QUOTE. When SEGMENT_FINE_SNAP is on we
locate the quote in supadata's per-word (interpolated) times to tighten the cut past
the ~chunk granularity; otherwise we use the chunk edge. Overlaps are trimmed
deterministically so no two clips share footage.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from pydantic import BaseModel

from .. import config
from ..llm import llm_json

ProgressCb = Optional[Callable[[float, str], None]]

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


# ── LLM schema ───────────────────────────────────────────────────────────────
class _Topic(BaseModel):
    title: str
    start_line: int
    end_line: int
    start_quote: str = ""      # first few words of the topic (for fine-snapping the start)
    end_quote: str = ""        # last few words of the topic (for fine-snapping the end)
    reason: str = ""
    facet: str = "other"


class _Plan(BaseModel):
    topics: list[_Topic]


def _mmss(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def _prompts(lines: str, n: int) -> "tuple[str, str]":
    system = (
        "You segment a lecture/talk transcript into self-contained CLIPS for a short-form "
        "learning feed. First read and understand the WHOLE transcript. Then split it into "
        "each SUBSTANTIVE topic — one coherent idea, concept, worked example, or section, "
        "taught from its introduction through to its natural conclusion. Skip pure filler "
        "(greetings, admin, 'like and subscribe', tangents).\n"
        "For every topic return: title; start_line (the line where the idea is INTRODUCED); "
        "end_line (the line where it CLOSES); start_quote (the first ~6 words spoken at the "
        "start, copied verbatim from that line); end_quote (the last ~6 words, verbatim); a "
        "short reason. Rules: (1) a clip must START at the beginning of the topic and END at "
        "its end — never mid-thought; (2) clips must NOT overlap — each line belongs to at "
        "most one clip; (3) go in chronological order; (4) line indices range from 0 to "
        f"{n - 1} — never exceed {n - 1}."
    )
    user = (
        f"Transcript ({n} lines, each formatted `[index] MM:SS text`):\n\n" + lines +
        "\n\nReturn every substantive topic as {title, start_line, end_line, start_quote, "
        "end_quote, reason, facet}."
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


# ── plan → clips ─────────────────────────────────────────────────────────────
def _plan_to_clips(plan: _Plan, segs: list[dict], words: list[dict],
                   settings: dict) -> list[dict]:
    n = len(segs)
    fine = settings.get("segment_fine_snap")
    fine = config.SEGMENT_FINE_SNAP if fine is None else bool(fine)
    min_s = float(settings.get("segment_min_clip_s") or config.SEGMENT_MIN_CLIP_S)
    tail_pad = float(settings.get("tail_pad_s", config.DEFAULTS["tail_pad_s"]))

    raw: list[dict] = []
    for tp in plan.topics:
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
        raw.append({"start": start, "end": end, "title": (tp.title or "").strip(),
                    "facet": (tp.facet or "other").strip() or "other",
                    "reason": (tp.reason or "").strip()})

    raw.sort(key=lambda c: (c["start"], c["end"]))
    # deterministic overlap trim: each clip starts strictly after the previous clip's end
    clips: list[dict] = []
    prev_end = -1.0
    for c in raw:
        if c["start"] < prev_end:
            c["start"] = prev_end
        if c["end"] - c["start"] < min_s:
            continue                                  # too short after trimming — drop
        c["cut_end"] = round(c["end"] + tail_pad, 3)
        c["start"] = round(c["start"], 3)
        c["end"] = round(c["end"], 3)
        prev_end = c["end"]
        clips.append(c)

    max_clips = int(settings.get("max_clips") or config.SEGMENT_MAX_CLIPS)
    clips = clips[:max_clips]
    for i, c in enumerate(clips):
        c["sequence_index"] = i + 1
    return clips


# ── public entry point ───────────────────────────────────────────────────────
def segment_clips(transcript: dict, settings: dict,
                  progress: ProgressCb = None) -> "tuple[list[dict], str]":
    """One Gemini comprehension pass → substantive topic clips. Returns (clips_spec, notes)."""
    segs = transcript.get("segments") or []
    words = transcript.get("words") or []
    if not segs:
        return [], "No transcript segments to segment."
    n = len(segs)
    lines = "\n".join(f"[{i}] {_mmss(s.get('start', 0.0))} {(s.get('text') or '').strip()}"
                      for i, s in enumerate(segs))
    system, user = _prompts(lines, n)
    model = settings.get("segment_model") or config.SEGMENT_MODEL

    if progress:
        progress(0.1, "Understanding the transcript…")
    plan = llm_json(system, user, _Plan, temperature=0.2, model=model,
                    max_output_tokens=config.SEGMENT_MAX_OUTPUT_TOKENS)
    if progress:
        progress(0.85, "Placing clip boundaries…")
    clips = _plan_to_clips(plan, segs, words, settings)
    if progress:
        progress(1.0, f"{len(clips)} clip(s) ready")
    notes = f"{len(clips)} topic clip(s) from {n} transcript segments (Gemini-segment engine)."
    return clips, notes
