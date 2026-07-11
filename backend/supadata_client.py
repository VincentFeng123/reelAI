"""Supadata transcript API client.

GET https://api.supadata.ai/v1/transcript?url=...&mode=auto  (header x-api-key)
Returns chunks [{text, offset(ms), duration(ms), lang}]. Long videos may return
HTTP 202 {jobId}; we poll /v1/transcript/{jobId} until it completes.

NOTE: uses httpx, not urllib — urllib.request capitalizes header names
("x-api-key" → "X-api-key"), which Supadata's auth rejects as missing.
"""
from __future__ import annotations

import time

import httpx

from . import config
from .errors import PipelineError


def _get(path: str, params: dict | None = None) -> tuple[int, dict]:
    if not config.SUPADATA_API_KEY:
        raise PipelineError(
            "SUPADATA_API_KEY is not set. Add it to .env (get a key at https://supadata.ai)."
        )
    headers = {"x-api-key": config.SUPADATA_API_KEY, "Accept": "application/json"}
    try:
        r = httpx.get(f"{config.SUPADATA_BASE}{path}", params=params, headers=headers, timeout=90.0)
    except httpx.RequestError as e:
        raise PipelineError(f"Could not reach Supadata: {e}")

    if r.status_code in (401, 403):
        raise PipelineError("Supadata API key is missing or invalid.")
    if r.status_code == 404:
        raise PipelineError("No transcript available for this video (Supadata 404).")
    if r.status_code == 429:
        raise PipelineError("Supadata rate limit/quota hit — try again shortly.")
    if r.status_code >= 400:
        raise PipelineError(f"Supadata error {r.status_code}: {r.text[:300]}")
    try:
        return r.status_code, r.json()
    except Exception:
        raise PipelineError("Supadata returned a non-JSON response.")


def _normalize(content) -> list[dict]:
    chunks: list[dict] = []
    for c in content or []:
        offset = float(c.get("offset", 0)) / 1000.0
        dur = float(c.get("duration", 0)) / 1000.0
        text = (c.get("text") or "").strip()
        if not text:
            continue
        chunks.append({"text": text, "start": offset, "end": offset + dur})
    return chunks


def fetch_transcript(url: str, lang: str = "en", chunk_size: int | None = None) -> list[dict]:
    """Return [{text, start(sec), end(sec)}] ordered by time."""
    params = {"url": url, "text": "false", "mode": "auto"}
    if lang:
        params["lang"] = lang
    if chunk_size:
        params["chunkSize"] = str(chunk_size)

    status, data = _get("/transcript", params)

    # Async job → poll
    if status == 202 or (isinstance(data, dict) and data.get("jobId") and "content" not in data):
        job_id = data["jobId"]
        for _ in range(60):  # up to ~3 min
            time.sleep(3)
            _, jd = _get(f"/transcript/{job_id}")
            state = (jd.get("status") or jd.get("state") or "").lower()
            if jd.get("content") is not None or state in ("completed", "succeeded", "done"):
                data = jd
                break
            if state in ("failed", "error", "cancelled"):
                raise PipelineError(f"Supadata transcript job failed: {jd.get('error', state)}")
        else:
            raise PipelineError("Supadata transcript timed out.")

    content = data.get("content") if isinstance(data, dict) else None
    if isinstance(content, str):
        raise PipelineError("Supadata returned plain text without timestamps.")
    chunks = _normalize(content)
    if not chunks:
        raise PipelineError("Supadata returned an empty transcript for this video.")
    return chunks
