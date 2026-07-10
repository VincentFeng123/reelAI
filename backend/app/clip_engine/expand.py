"""Topic → diverse YouTube search queries. Gemini when keyed, else a keyless
deterministic fallback. Port of practice/lib/expand.js (LLM path + free path).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

from . import config
from .cancellation import raise_if_cancelled, run_cancellable
from .errors import CancellationError

_SYSTEM = (
    "You expand a user's study topic into diverse YouTube search queries that surface "
    "EDUCATIONAL content — lectures, explainers, university courses, tutorials, and "
    "documentaries. Spellcheck and correct the input, infer the academic field or discipline "
    "(e.g. ambiguous 'jaguar' → 'jaguar animal biology'), then produce up to N distinct queries "
    "using phrasings like 'X explained', 'X lecture', 'how X works', 'X course', 'X for "
    "students', 'introduction to X', 'X fundamentals', and field-qualified variants. "
    "AVOID entertainment phrasings such as reactions, memes, funny compilations, top-10 lists, "
    "vlogs, or challenge videos — this is a study product. "
    "Return ONLY strict JSON: {\"corrected\": \"...\", \"queries\": [\"q1\", ...]} with the "
    "corrected topic first in queries. No prose, no code fences."
)


def _user(topic: str, n: int) -> str:
    return (f"User topic: {json.dumps(topic)}\nN = {n}\n"
            f"Return JSON with \"corrected\" and \"queries\" (at most {n}, corrected first). JSON only.")


def _safe_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except Exception:
        return None


def _normalize(corrected: str | None, queries, fallback: str, n: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def push(q):
        if not q:
            return
        s = str(q).strip()
        if not s or s.lower() in seen:
            return
        seen.add(s.lower())
        out.append(s)

    push(corrected or fallback)
    for q in (queries or []):
        push(q)
    if not out:
        push(fallback)
    return out[:n]


def free_expand(topic: str, n: int) -> dict:
    variants = [
        topic,
        f"{topic} explained",
        f"{topic} lecture",
        f"{topic} tutorial",
        f"how {topic} works",
        f"{topic} course",
        f"{topic} for beginners",
    ]
    return {"corrected": topic, "queries": _normalize(topic, variants, topic, n),
            "provider_used": "free"}


async def _gemini_expand_raw_async(system: str, user: str, model: str) -> str:
    from google import genai  # lazy import
    client = genai.Client(api_key=config.require_gemini_key())
    resp = await client.aio.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": system + "\n\n" + user}]}],
        config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    return getattr(resp, "text", "") or ""


def _gemini_expand_raw(
    system: str,
    user: str,
    model: str,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    return run_cancellable(
        lambda: _gemini_expand_raw_async(system, user, model), should_cancel
    )


_LEVEL_LINES = {
    "beginner": (
        " The viewer is a BEGINNER on this topic: prefer phrasings like "
        "'introduction to X', 'X basics', 'X for beginners', 'X crash course'; "
        "avoid graduate-level or research phrasings."
    ),
    "advanced": (
        " The viewer is ADVANCED on this topic: prefer phrasings like "
        "'advanced X', 'graduate X lecture', 'X deep dive', 'X seminar'; "
        "avoid 'for beginners' phrasings."
    ),
}


def expand_query(
    topic: str,
    n: int,
    level: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    raise_if_cancelled(should_cancel)
    topic = topic.strip()
    system = _SYSTEM + _LEVEL_LINES.get((level or "").strip().lower(), "")
    if not config.GEMINI_API_KEY:
        return free_expand(topic, n)
    try:
        raw = (
            _gemini_expand_raw(system, _user(topic, n), config.EXPAND_MODEL)
            if should_cancel is None
            else _gemini_expand_raw(
                system, _user(topic, n), config.EXPAND_MODEL, should_cancel
            )
        )
        parsed = _safe_json(raw)
        if parsed:
            corrected = parsed.get("corrected") or topic
            queries = _normalize(corrected, parsed.get("queries"), topic, n)
            if len(queries) < n:
                queries = _normalize(corrected, [*queries, *free_expand(corrected, n)["queries"]], topic, n)
            return {"corrected": corrected, "queries": queries, "provider_used": "gemini"}
    except CancellationError:
        raise
    except Exception:
        pass
    return free_expand(topic, n)
