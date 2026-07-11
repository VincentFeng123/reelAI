"""The punctuation-restoration prompt: system rules, compact token rendering, repair prompts.

``PROMPT_VERSION`` is folded into the cache key so a prompt change invalidates cached results.
"""
from __future__ import annotations

import json

from .types import TimedWord, TranscriptChunk

PROMPT_VERSION = "punct-v1"

SYSTEM_PROMPT = (
    "You are restoring punctuation and capitalization to an automatic speech-recognition "
    "transcript.\n\n"
    "You will receive an ordered list of immutable transcript tokens. Each token has an ID, "
    "original word, timing information, and optionally a speaker.\n\n"
    "Your task is to annotate the existing tokens.\n\n"
    "Rules:\n"
    "1. Do not add words.\n"
    "2. Do not delete words.\n"
    "3. Do not replace words.\n"
    "4. Do not reorder words.\n"
    "5. Do not correct grammar.\n"
    "6. Do not correct factual errors.\n"
    "7. Do not paraphrase.\n"
    "8. Do not combine or split tokens.\n"
    "9. Add only capitalization and punctuation annotations.\n"
    "10. Identify sentence boundaries.\n"
    "11. Identify paragraph boundaries only when there is a strong topic or speaker transition.\n"
    "12. Use question marks only when the wording genuinely forms a question.\n"
    "13. Preserve technical terminology exactly as provided.\n"
    "14. Preserve filler words unless the original transcript pipeline removes them elsewhere.\n"
    "15. Use timing gaps and speaker changes as supporting evidence, not absolute rules.\n"
    "16. Return valid structured JSON matching the provided schema.\n"
    "17. Return exactly one annotation for every input token ID that needs one.\n"
    "18. Never return an annotation for a token ID that was not provided.\n\n"
    "Output format — return ONLY the tokens that change (a SPARSE list). For each such token "
    "emit an object: {\"id\": <tokenId>, \"p\": <punctuationAfter>, \"cap\": <capitalize>, "
    "\"ss\": <sentenceStart>, \"se\": <sentenceEnd>, \"pg\": <paragraphStart>, \"conf\": <0..1>}. "
    "Fields default to no-punctuation / not-capitalized / mid-sentence, so omit tokens that keep "
    "the defaults. 'p' must be one of \"\" \".\" \",\" \"?\" \"!\" \":\" \";\" (no quotation marks). "
    "'cap' capitalizes the first letter of that token. Mark the LAST token of every sentence with "
    "se=true and a terminal 'p' ('.', '?' or '!'); mark the FIRST token of every sentence with "
    "ss=true and cap=true."
)

STRICT_REPAIR_SYSTEM = (
    SYSTEM_PROMPT
    + "\n\nSTRICT REPAIR MODE: your previous replies were invalid. Return ONLY minimal, correct "
    "sparse edits for the SAME token IDs listed below. Do not include any id that is not in the "
    "list. Every sentence must have exactly one first token (ss=true, cap=true) and one last "
    "token (se=true, terminal 'p'). Output JSON only, no prose."
)


def _pause_ms(prev_end: float | None, cur_start: float, cur_end: float,
             next_start: float | None) -> tuple[int, int]:
    before = 0 if prev_end is None else max(0, round((cur_start - prev_end) * 1000))
    after = 0 if next_start is None else max(0, round((next_start - cur_end) * 1000))
    return before, after


def render_tokens(chunk: TranscriptChunk, words: list[TimedWord]) -> str:
    """Compact JSON of the chunk's tokens. Omits null metadata to keep the prompt small."""
    ids = chunk.token_ids
    items: list[dict] = []
    for pos, gi in enumerate(ids):
        w = words[gi]
        prev_end = words[ids[pos - 1]].end if pos > 0 else None
        next_start = words[ids[pos + 1]].start if pos + 1 < len(ids) else None
        before, after = _pause_ms(prev_end, w.start, w.end, next_start)
        item: dict = {"id": w.id, "word": w.word.strip()}
        if before:
            item["pauseBeforeMs"] = before
        if after:
            item["pauseAfterMs"] = after
        if w.speaker:
            item["speaker"] = w.speaker
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(chunk: TranscriptChunk, words: list[TimedWord]) -> str:
    return (
        "Annotate these transcript tokens. Return only the sparse edits described in the system "
        "rules — exactly the token IDs given, none others.\n\nTOKENS:\n"
        + render_tokens(chunk, words)
    )


def repair_prompt(chunk: TranscriptChunk, words: list[TimedWord], errors: list[str]) -> str:
    return (
        build_user_prompt(chunk, words)
        + "\n\nYour previous answer failed these checks:\n"
        + "\n".join(f"- {e}" for e in errors[:12])
        + "\nReturn corrected sparse edits for the SAME token IDs only."
    )


def strict_repair_prompt(chunk: TranscriptChunk, words: list[TimedWord]) -> str:
    valid_ids = ", ".join(words[gi].id for gi in chunk.token_ids)
    return (
        "Valid token IDs (use only these):\n"
        + valid_ids
        + "\n\nTOKENS:\n"
        + render_tokens(chunk, words)
        + "\n\nReturn minimal correct sparse edits (JSON only)."
    )
