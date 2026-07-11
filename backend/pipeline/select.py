"""AI segment selection — informative, self-contained clips WITH CONTEXT.

The model reads a timestamped transcript and returns complete segment time ranges
(it decides boundaries, so a worked example includes the problem being read, the
analysis, AND the solution — not just a fragment). We then anchor those to real
transcript boundaries and drop overlaps by score. Approach adapted from the
AI-Youtube-Shorts-Generator highlight selector, tuned for informativeness + context.
"""
from __future__ import annotations

import json
import math
import re
from typing import Callable, Optional

from pydantic import BaseModel, ValidationError

from .. import config
from .sentences import Sentence

ProgressCb = Optional[Callable[[float, str], None]]

# ── tokens / rendering ───────────────────────────────────────────────────────
_tok_enc = None


def estimate_tokens(text: str) -> int:
    global _tok_enc
    try:
        if _tok_enc is None:
            import tiktoken
            _tok_enc = tiktoken.get_encoding("o200k_base")
        return len(_tok_enc.encode(text))
    except Exception:
        return math.ceil(len(text) / config.CHARS_PER_TOKEN)


def render_sentences(sents: list[Sentence]) -> str:
    return "\n".join(f"[{s.idx}] ({s.start:.1f}-{s.end:.1f}) {s.text}" for s in sents)


# ── schema / models ──────────────────────────────────────────────────────────
SELECTION_SCHEMA = {
    "name": "segment_selection",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                        "facet": {"type": "string"},
                        "title": {"type": "string"},
                        "reason": {"type": "string"},
                        "score": {"type": "integer"},
                        "quote_start": {"type": "string"},
                        "quote_end": {"type": "string"},
                    },
                    "required": ["start_time", "end_time", "facet", "title", "reason",
                                 "score", "quote_start", "quote_end"],
                },
            },
            "topic_present": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": ["segments", "topic_present", "notes"],
    },
}


class Segment(BaseModel):
    start_time: float = 0.0
    end_time: float = 0.0
    facet: str = "other"
    title: str = ""
    reason: str = ""
    score: int = 0
    quote_start: str = ""
    quote_end: str = ""


class SelectionResult(BaseModel):
    segments: list[Segment] = []
    topic_present: bool = True
    notes: str = ""


SYSTEM_PROMPT = """You are an expert at finding the most INFORMATIVE, self-contained segments of a video about a given TOPIC and clipping each one WITH FULL CONTEXT.

You are given the TOPIC and a timestamped transcript. Each line is: [index] (start-end seconds) text.

What makes a great clip:
- It teaches or conveys ONE complete idea about the topic and works as a standalone clip.
- It includes the necessary CONTEXT, not just the punchline. For a worked example or problem, the clip must START where the problem is introduced/read and the setup or analysis begins, and END after the result is reached and explained. NEVER start at the answer or cut off the setup.
- It begins at a natural starting point (the start of a thought/sentence) and ends at a natural stopping point (a completed thought). Never start or end mid-sentence or mid-thought.

Rules:
1. Select only segments genuinely ABOUT the topic, and the most informative ones.
2. Separate DISTINCT points into separate clips: a definition, a derivation, EACH distinct worked example/problem, an application, an intuition, a comparison. Do not merge unrelated points; do not return near-duplicates.
3. Each clip should usually be 45-150 seconds — long enough to include full context, short enough to stay focused. Use up to ~180 seconds only when a complete worked problem genuinely needs it. NEVER return the whole video or a multi-minute span that spills across many unrelated topics.
4. Decide the NUMBER of clips from the content: few or zero if the topic barely appears, more if it is richly covered. Never pad with weak segments.
5. For each clip return:
   - start_time, end_time: seconds, taken from the transcript timestamps, covering the COMPLETE segment WITH its context.
   - facet: one of definition, derivation, worked_example, application, intuition, comparison, overview, other.
   - title: a short, specific title.
   - reason: one sentence on what makes it informative and self-contained (mention the context it includes).
   - score: 0-100, how informative and clip-worthy it is about the topic.
   - quote_start: the FIRST ~10 words of the clip's first sentence, copied verbatim from the transcript.
   - quote_end: the LAST ~10 words of the clip's last sentence, copied verbatim from the transcript.
6. Also return topic_present (boolean) and a short notes string. If the topic barely appears, set topic_present=false, return an empty segments array, and explain in notes.

First infer what kind of content this is (lecture, tutorial, problem-solving, talk, etc.), then select. Output ONLY the structured result."""

JSON_OBJECT_INSTRUCTION = (
    'Respond with ONLY a JSON object of EXACTLY this shape — no prose, no markdown fences:\n'
    '{"segments":[{"start_time":<number>,"end_time":<number>,"facet":<string>,"title":<string>,'
    '"reason":<string>,"score":<integer>,"quote_start":<string>,"quote_end":<string>}],'
    '"topic_present":<boolean>,"notes":<string>}'
)


def build_user_prompt(topic: str, rendered: str) -> str:
    return (
        f"TOPIC: {topic}\n\n"
        f"TRANSCRIPT (each line: [index] (start-end seconds) text):\n{rendered}\n\n"
        "Select the distinct, complete, maximally-informative segments about the TOPIC, "
        "each WITH full context. Decide the number from the content."
    )


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def run_llm_selection(sents: list[Sentence], topic: str) -> SelectionResult:
    user = build_user_prompt(topic, render_sentences(sents))
    est = estimate_tokens(SYSTEM_PROMPT + user) + config.EXPECTED_OUTPUT_TOKENS

    if config.LLM_PROVIDER == "gemini":
        from .. import gemini_client
        raw = gemini_client.generate_json(SYSTEM_PROMPT, user, SelectionResult)
        try:
            return SelectionResult.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError, ValueError):
            raw = gemini_client.generate_json(
                SYSTEM_PROMPT, user + "\n\n" + JSON_OBJECT_INSTRUCTION, SelectionResult
            )
            return SelectionResult.model_validate_json(_strip_fences(raw))

    # GROQ provider
    from ..groq_client import chat
    try:
        raw = chat(config.LLM_PRIMARY, SYSTEM_PROMPT, user,
                   response_format={"type": "json_schema", "json_schema": SELECTION_SCHEMA},
                   est_tokens=est)
        return SelectionResult.model_validate_json(raw)
    except Exception:
        pass
    fb = user + "\n\n" + JSON_OBJECT_INSTRUCTION
    last: Optional[Exception] = None
    for _ in range(config.FALLBACK_RETRIES + 1):
        raw = chat(config.LLM_FALLBACK, SYSTEM_PROMPT, fb,
                   response_format={"type": "json_object"}, est_tokens=est)
        try:
            return SelectionResult.model_validate_json(_strip_fences(raw))
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            last = e
            fb = user + "\n\n" + JSON_OBJECT_INSTRUCTION + f"\n\nPrevious reply invalid ({str(e)[:120]}). Return corrected JSON only."
    raise RuntimeError(f"LLM selection failed: {last}")


# ── quote anchoring (recover clean boundaries) ───────────────────────────────
def _normalize(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def _nearest_by_time(sents: list[Sentence], t: float) -> int:
    return min(range(len(sents)),
               key=lambda i: min(abs(sents[i].start - t), abs(sents[i].end - t)))


def anchor_quote(quote: str, sents: list[Sentence], approx_time: float) -> int:
    from rapidfuzz import fuzz
    q = _normalize(quote)
    if not q:
        return _nearest_by_time(sents, approx_time)
    best_idx, best_adj, best_raw = None, -1e9, 0.0
    for s in sents:
        raw = fuzz.partial_ratio(q, _normalize(s.text))
        dt = max(0.0, min(abs(s.start - approx_time), abs(s.end - approx_time)) - config.QUOTE_TIME_WINDOW)
        adj = raw - dt * config.QUOTE_TIME_PENALTY
        if adj > best_adj:
            best_idx, best_adj, best_raw = s.idx, adj, raw
    if best_raw < config.MIN_QUOTE_SCORE:
        return _nearest_by_time(sents, approx_time)
    return best_idx if best_idx is not None else _nearest_by_time(sents, approx_time)


def to_candidate(seg: Segment, sents: list[Sentence], by_idx: dict[int, Sentence]) -> Optional[dict]:
    # anchor by quotes; fall back to the model's emitted times
    i0 = anchor_quote(seg.quote_start, sents, seg.start_time)
    i1 = anchor_quote(seg.quote_end, sents, seg.end_time)
    if i1 < i0:
        i0, i1 = sorted((i0, i1))
    if i0 not in by_idx or i1 not in by_idx:
        return None
    text = " ".join(by_idx[i].text for i in range(i0, i1 + 1) if i in by_idx)
    return {
        "i_start": i0, "i_end": i1,
        "start": by_idx[i0].start, "end": by_idx[i1].end,
        "text": text, "facet": seg.facet or "other", "title": seg.title,
        "reason": seg.reason, "score": int(seg.score), "rel": float(seg.score) / 100.0,
    }


# ── overlap dedupe (score-based, like the reference) ─────────────────────────
def _dedupe_overlap(cands: list[dict]) -> list[dict]:
    cands = sorted(cands, key=lambda c: c.get("score", 0), reverse=True)
    kept: list[dict] = []
    for c in cands:
        s, e = c["start"], c["end"]
        dur = max(1e-6, e - s)
        clash = False
        for k in kept:
            overlap = min(e, k["end"]) - max(s, k["start"])
            if overlap > 0 and overlap > 0.5 * min(dur, k["end"] - k["start"]):
                clash = True
                break
        if not clash:
            kept.append(c)
    return kept


# ── orchestrator ─────────────────────────────────────────────────────────────
def select_segments(sentences: list[Sentence], topic: str, settings: dict,
                    progress: ProgressCb = None) -> tuple[list[dict], str]:
    if not sentences:
        return [], "Transcript was empty."
    by_idx = {s.idx: s for s in sentences}

    def emit(frac, msg=""):
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    emit(0.2, "Reading the transcript…")
    result = run_llm_selection(sentences, topic)
    if not result.topic_present or not result.segments:
        return [], result.notes or "No segments about this topic were found."

    emit(0.7, "Locating clip boundaries…")
    cands = [c for seg in result.segments if (c := to_candidate(seg, sentences, by_idx))]
    if not cands:
        return [], result.notes or "Could not locate the selected segments."

    cands = _dedupe_overlap(cands)
    cands.sort(key=lambda c: c["start"])
    emit(1.0, f"Selected {len(cands)} segment(s)")
    return cands, result.notes
