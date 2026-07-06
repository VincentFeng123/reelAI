"""Groq SDK wrapper: transcription + chat, with a proactive token-budget limiter
and header-aware 429/5xx backoff.

The Groq client targets api.groq.com by default (OpenAI-compatible); we don't
override base_url. All free-tier limits live in config.py.
"""
from __future__ import annotations

import os
import random
import re
import threading
import time
from collections import deque
from typing import Callable, Optional

from . import config

# Fallback exception stubs — replaced with the real groq classes inside get_client()
# when groq is actually installed. On the default path (supadata + gemini), groq
# is never imported and these stubs are never instantiated.
class RateLimitError(Exception): ...
class APIStatusError(Exception):
    status_code = 500
class BadRequestError(Exception): ...

_client = None
_client_lock = threading.Lock()


def get_client():
    global _client, RateLimitError, APIStatusError, BadRequestError
    if _client is None:
        with _client_lock:
            if _client is None:
                from groq import Groq  # lazy: only needed when actually using Groq
                try:
                    from groq import (  # noqa: F401
                        APIStatusError as _ASE,
                        BadRequestError as _BRE,
                        RateLimitError as _RLE,
                    )
                    APIStatusError, BadRequestError, RateLimitError = _ASE, _BRE, _RLE
                except Exception:
                    pass
                if not config.GROQ_API_KEY:
                    raise RuntimeError(
                        "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                        "free key from https://console.groq.com"
                    )
                _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


# ── Backoff ─────────────────────────────────────────────────────────────────
def parse_groq_duration(s: Optional[str]) -> Optional[float]:
    """Parse Groq duration strings: '2m59.56s', '7.66s', '880ms', '5'."""
    if not s:
        return None
    s = s.strip()
    if s.endswith("ms"):
        try:
            return float(s[:-2]) / 1000.0
        except ValueError:
            return None
    total = 0.0
    for val, unit in re.findall(r"([\d.]+)\s*([smh])", s):
        total += float(val) * {"s": 1, "m": 60, "h": 3600}[unit]
    if total:
        return total
    try:
        return float(s)  # bare seconds (Retry-After integer)
    except ValueError:
        return None


def _retry_after(e: Exception) -> Optional[float]:
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        w = parse_groq_duration(headers.get(key))
        if w is not None:
            return w
    return None


def with_backoff(call: Callable, label: str = "groq"):
    for attempt in range(config.BACKOFF_MAX_RETRIES + 1):
        try:
            return call()
        except RateLimitError as e:  # HTTP 429
            if attempt == config.BACKOFF_MAX_RETRIES:
                raise
            wait = _retry_after(e) or min(
                config.BACKOFF_CAP, config.BACKOFF_BASE * 2 ** attempt
            )
            time.sleep(wait + random.uniform(0, 0.5 * max(wait, 0.1)))
        except APIStatusError as e:  # 5xx → retry; 4xx → raise
            status = getattr(e, "status_code", 500)
            if status < 500 or attempt == config.BACKOFF_MAX_RETRIES:
                raise
            time.sleep(
                min(config.BACKOFF_CAP, config.BACKOFF_BASE * 2 ** attempt)
                + random.uniform(0, 1.0)
            )


# ── Proactive token budget (trailing 60 s rolling window) ───────────────────
class RateBudget:
    def __init__(self) -> None:
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, est_tokens: int) -> None:
        while True:
            with self._lock:
                now = time.time()
                while self._events and now - self._events[0][0] > 60:
                    self._events.popleft()
                used = sum(t for _, t in self._events)
                reqs = len(self._events)
                if (
                    used + est_tokens <= config.TPM_LIMIT * config.TPM_SAFETY
                    and reqs + 1 <= config.RPM_LIMIT
                ):
                    self._events.append((now, est_tokens))
                    return
            time.sleep(0.5)


budget = RateBudget()


# ── Transcription ───────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str, language: str = "en") -> dict:
    """Return {'text', 'duration', 'words':[{word,start,end}], 'segments':[{start,end,text}]}."""
    client = get_client()
    with open(audio_path, "rb") as f:
        data = f.read()

    def _call():
        return client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), data),
            model=config.STT_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=language,
            temperature=0.0,
        )

    result = with_backoff(_call, label="transcribe")
    payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return {
        "text": payload.get("text", ""),
        "duration": payload.get("duration"),
        "words": payload.get("words") or [],
        "segments": payload.get("segments") or [],
    }


# ── Chat ────────────────────────────────────────────────────────────────────
def chat(
    model: str,
    system: str,
    user: str,
    response_format: Optional[dict] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    est_tokens: int = 0,
) -> str:
    """Single chat completion. Returns the assistant message content string."""
    client = get_client()
    if est_tokens:
        budget.acquire(est_tokens)
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = with_backoff(lambda: client.chat.completions.create(**kwargs), label="chat")
    return resp.choices[0].message.content or ""
