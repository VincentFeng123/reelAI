"""Topic → diverse YouTube search queries. Gemini when keyed, else a keyless
deterministic fallback. Port of practice/lib/expand.js (LLM path + free path).
"""
from __future__ import annotations

import json
import re

from . import config

_SYSTEM = (
    "You expand a user's search topic into a diverse set of YouTube search queries that "
    "maximize topical coverage. Spellcheck and correct the input, infer intent, then produce "
    "up to N distinct queries covering the corrected topic, close synonyms, important "
    "sub-topics, and phrase variants (\"X tutorial\", \"X explained\", \"X for beginners\"). "
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
    variants = [topic, f"{topic} explained", f"{topic} tutorial", f"{topic} for beginners"]
    return {"corrected": topic, "queries": _normalize(topic, variants, topic, n),
            "provider_used": "free"}


def _gemini_expand_raw(system: str, user: str, model: str) -> str:
    from google import genai  # lazy import
    client = genai.Client(api_key=config.require_gemini_key())
    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": system + "\n\n" + user}]}],
        config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    return getattr(resp, "text", "") or ""


def expand_query(topic: str, n: int) -> dict:
    topic = topic.strip()
    if not config.GEMINI_API_KEY:
        return free_expand(topic, n)
    try:
        raw = _gemini_expand_raw(_SYSTEM, _user(topic, n), config.GEMINI_MODEL)
        parsed = _safe_json(raw)
        if parsed:
            return {"corrected": parsed.get("corrected") or topic,
                    "queries": _normalize(parsed.get("corrected"), parsed.get("queries"), topic, n),
                    "provider_used": "gemini"}
    except Exception:
        pass
    return free_expand(topic, n)
