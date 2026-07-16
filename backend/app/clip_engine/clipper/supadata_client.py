"""Supadata timed-transcript client backed by validated transcript artifacts."""
from __future__ import annotations

import html
import math
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from . import config
from ..cancellation import raise_if_cancelled, run_cancellable, sleep_with_probe
from ..errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTransientError,
    TranscriptUnavailableError,
)
from ..metadata import normalize_youtube_video_id
from ..provider_cache import (
    DEFAULT_PROVIDER_CACHE,
    TRANSCRIPT_SCHEMA_VERSION,
    ProviderCacheStore,
    TranscriptArtifact,
    normalize_language,
    transcript_artifact_key,
    validate_transcript_payload,
)
from ..provider_runtime import (
    GenerationContext,
    MAX_PROVIDER_RETRIES,
    bounded_retry_after,
)

DEFAULT_TRANSCRIPT_DEADLINE_SEC = 180.0
TRANSCRIPT_POLL_INTERVAL_SEC = 1.0


def _normalized_caption_text(value: object) -> str:
    text = str(value or "")
    # Some generated YouTube captions arrive double-escaped (for example,
    # ``we&amp;#39;ll``). Decode at the provider boundary so quote grounding,
    # dangling-clause guards, captions, and persisted snippets all see the same
    # spoken text.
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return " ".join(text.split()).strip()


def _deadline_or_default(value: float | None) -> float:
    try:
        parsed = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        parsed = float("nan")
    if math.isfinite(parsed):
        return parsed
    return time.monotonic() + DEFAULT_TRANSCRIPT_DEADLINE_SEC


def _remaining_seconds(deadline_monotonic: float) -> float:
    return max(0.0, deadline_monotonic - time.monotonic())


def _raise_if_deadline_exceeded(deadline_monotonic: float) -> None:
    if _remaining_seconds(deadline_monotonic) <= 0:
        raise ProviderTransientError(
            "Supadata transcript retrieval timed out.",
            provider="supadata",
            operation="transcript",
            detail="generation deadline exceeded",
        )


async def _sleep_before_retry(
    seconds: float,
    *,
    should_cancel: Callable[[], bool] | None,
    deadline_monotonic: float,
) -> None:
    _raise_if_deadline_exceeded(deadline_monotonic)
    await sleep_with_probe(
        min(max(0.0, seconds), _remaining_seconds(deadline_monotonic)),
        should_cancel,
    )
    _raise_if_deadline_exceeded(deadline_monotonic)


def _error_detail(response: httpx.Response) -> str:
    try:
        value = response.json()
    except Exception:
        return str(getattr(response, "text", "") or "")[:300]
    if isinstance(value, dict):
        return str(value.get("message") or value.get("error") or "")[:300]
    return ""


def _failure(
    response: httpx.Response,
    *,
    retry_after: float | None = None,
) -> ProviderError:
    status = int(response.status_code)
    kwargs = dict(
        provider="supadata",
        operation="transcript",
        status_code=status,
        detail=_error_detail(response) or None,
    )
    if status in (401, 403):
        return ProviderAuthenticationError("Supadata authentication failed.", **kwargs)
    if status == 402:
        return ProviderQuotaError("Supadata quota is exhausted.", **kwargs)
    if status == 404:
        return TranscriptUnavailableError(
            "No usable timestamped transcript is available for this video.", **kwargs
        )
    if status == 429:
        return ProviderRateLimitError(
            "Supadata transcript retrieval is rate limited.",
            retry_after_sec=retry_after,
            **kwargs,
        )
    if 500 <= status <= 599:
        return ProviderTransientError("Supadata transcript retrieval is unavailable.", **kwargs)
    return ProviderRequestError(f"Supadata rejected the transcript request ({status}).", **kwargs)


async def _request(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, Any] | None,
    api_key: str,
    should_cancel: Callable[[], bool] | None,
    context: GenerationContext | None,
    reserve_budget: bool,
    deadline_monotonic: float,
) -> tuple[int, dict[str, Any]]:
    for retry_index in range(MAX_PROVIDER_RETRIES + 1):
        raise_if_cancelled(should_cancel)
        _raise_if_deadline_exceeded(deadline_monotonic)
        attempt = retry_index + 1
        if reserve_budget and context is not None:
            context.reserve("transcript")
        try:
            response = await client.get(
                f"{config.SUPADATA_BASE}{path}",
                params=params,
                headers={"x-api-key": api_key, "Accept": "application/json"},
                timeout=max(0.001, min(90.0, _remaining_seconds(deadline_monotonic))),
            )
        except httpx.RequestError as exc:
            if context is not None:
                context.record_http(
                    provider="supadata",
                    operation="transcript",
                    attempt=attempt,
                    status_code=None,
                    error_code="provider_transient",
                    metadata={"poll": not reserve_budget},
                )
            if retry_index < MAX_PROVIDER_RETRIES:
                await _sleep_before_retry(
                    min(30.0, 1.2 * attempt),
                    should_cancel=should_cancel,
                    deadline_monotonic=deadline_monotonic,
                )
                continue
            raise ProviderTransientError(
                "Could not reach Supadata transcript retrieval.",
                provider="supadata",
                operation="transcript",
                detail=str(exc),
            ) from exc

        status = int(response.status_code)
        error_code = ""
        if status == 429:
            error_code = "provider_rate_limited"
        elif status >= 500:
            error_code = "provider_transient"
        elif status >= 400:
            error_code = _failure(response).code
        if context is not None:
            context.record_http(
                provider="supadata",
                operation="transcript",
                attempt=attempt,
                status_code=status,
                headers=response.headers,
                error_code=error_code,
                metadata={"poll": not reserve_budget},
            )
        if status == 429 or 500 <= status <= 599:
            retry_after = bounded_retry_after(response.headers)
            if retry_index < MAX_PROVIDER_RETRIES:
                await _sleep_before_retry(
                    retry_after if retry_after is not None else min(30.0, 1.2 * attempt),
                    should_cancel=should_cancel,
                    deadline_monotonic=deadline_monotonic,
                )
                continue
            raise _failure(response, retry_after=retry_after)
        if status >= 400:
            raise _failure(response)
        try:
            data = response.json()
        except Exception as exc:
            raise ProviderRequestError(
                "Supadata returned invalid transcript JSON.",
                provider="supadata",
                operation="transcript",
                status_code=status,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderRequestError(
                "Supadata returned an invalid transcript payload.",
                provider="supadata",
                operation="transcript",
                status_code=status,
            )
        return status, data
    raise AssertionError("unreachable")


def _normalize_content(
    content: Any,
    *,
    video_id: str,
    returned_language: str,
) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    cues: list[dict[str, Any]] = []
    for index, raw in enumerate(content):
        if not isinstance(raw, dict):
            return []
        try:
            offset_ms = float(raw.get("offset"))
            duration_ms = float(raw.get("duration"))
        except (TypeError, ValueError):
            return []
        text = _normalized_caption_text(raw.get("text"))
        if not text:
            continue
        cues.append(
            {
                "cue_id": str(raw.get("id") or f"{video_id}:cue:{index}"),
                "text": text,
                "start": offset_ms / 1000.0,
                "end": (offset_ms + duration_ms) / 1000.0,
                "lang": normalize_language(str(raw.get("lang") or returned_language))
                or returned_language,
            }
        )
    return cues


async def _fetch_transcript_artifact_async(
    url: str,
    lang: str,
    should_cancel: Callable[[], bool] | None,
    *,
    chunk_size: int | None,
    context: GenerationContext | None,
    cache_store: ProviderCacheStore | None,
    deadline_monotonic: float | None,
) -> TranscriptArtifact:
    effective_deadline = _deadline_or_default(deadline_monotonic)
    _raise_if_deadline_exceeded(effective_deadline)
    video_id = normalize_youtube_video_id(url)
    if video_id is None:
        raise ProviderRequestError(
            "A canonical YouTube video URL or id is required for transcript retrieval.",
            provider="supadata",
            operation="transcript",
        )
    store = cache_store or (context.cache_store if context is not None else None) or DEFAULT_PROVIDER_CACHE
    if store.is_video_tombstoned(video_id):
        raise ProviderRequestError(
            "This YouTube video is blocked and cannot be retrieved.",
            provider="supadata",
            operation="transcript",
        )
    requested_language = normalize_language(lang) or "en"
    # Native-only artifacts from earlier runs are the fastest and most precise
    # cache hit. Auto-mode artifacts are the caption-optional fallback.
    for native_mode in (True, False):
        cached = store.get_transcript(
            video_id=video_id,
            provider="supadata",
            requested_language=requested_language,
            native_mode=native_mode,
            schema_version=TRANSCRIPT_SCHEMA_VERSION,
        )
        if cached is not None:
            if context is not None:
                context.record_cache_hit(
                    provider="supadata",
                    operation="transcript",
                    metadata={
                        "artifact_key": cached.artifact_key,
                        "native_mode": cached.native_mode,
                    },
                )
            return cached
    api_key = str(config.SUPADATA_API_KEY or "").strip()
    if not api_key:
        from ..config import require_supadata_key

        api_key = require_supadata_key()

    params = {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "text": "false",
        "mode": "auto",
        "lang": requested_language,
    }
    if chunk_size:
        params["chunkSize"] = str(max(1, int(chunk_size)))
    async with httpx.AsyncClient(
        timeout=max(0.001, min(90.0, _remaining_seconds(effective_deadline)))
    ) as client:
        status, data = await _request(
            client,
            "/transcript",
            params=params,
            api_key=api_key,
            should_cancel=should_cancel,
            context=context,
            reserve_budget=True,
            deadline_monotonic=effective_deadline,
        )
        if status == 202 or (data.get("jobId") and "content" not in data):
            job_id = str(data.get("jobId") or "").strip()
            if not job_id:
                raise ProviderRequestError(
                    "Supadata returned a transcript job without an id.",
                    provider="supadata",
                    operation="transcript",
                )
            while _remaining_seconds(effective_deadline) > 0:
                await _sleep_before_retry(
                    TRANSCRIPT_POLL_INTERVAL_SEC,
                    should_cancel=should_cancel,
                    deadline_monotonic=effective_deadline,
                )
                _, polled = await _request(
                    client,
                    f"/transcript/{job_id}",
                    params=None,
                    api_key=api_key,
                    should_cancel=should_cancel,
                    context=context,
                    reserve_budget=False,
                    deadline_monotonic=effective_deadline,
                )
                state = str(polled.get("status") or polled.get("state") or "").lower()
                result_payload = polled.get("result")
                completed_payload = result_payload if isinstance(result_payload, dict) else polled
                if completed_payload.get("content") is not None or state in {"completed", "succeeded", "done"}:
                    data = completed_payload
                    break
                if state in {"failed", "error", "cancelled"}:
                    raise TranscriptUnavailableError(
                        "Supadata could not prepare a usable timestamped transcript for this video.",
                        provider="supadata",
                        operation="transcript",
                        detail=str(polled.get("error") or state),
                    )
            else:
                _raise_if_deadline_exceeded(effective_deadline)

    content = data.get("content")
    if isinstance(content, str):
        raise TranscriptUnavailableError(
            "Supadata did not return timestamped transcript cues.",
            provider="supadata",
            operation="transcript",
        )
    returned_language = normalize_language(
        str(data.get("lang") or data.get("language") or "")
    )
    if not returned_language and isinstance(content, list):
        returned_language = next(
            (
                normalize_language(str(cue.get("lang") or ""))
                for cue in content
                if isinstance(cue, dict) and cue.get("lang")
            ),
            "",
        )
    returned_language = returned_language or requested_language
    cues = _normalize_content(
        content,
        video_id=video_id,
        returned_language=returned_language,
    )
    if not cues:
        raise TranscriptUnavailableError(
            "Supadata returned no usable timestamped transcript cues.",
            provider="supadata",
            operation="transcript",
        )
    duration_sec = float(cues[-1]["end"])
    created_at = datetime.now(timezone.utc).isoformat()
    artifact = validate_transcript_payload(
        {
            "artifact_key": transcript_artifact_key(
                video_id=video_id,
                provider="supadata",
                requested_language=requested_language,
                returned_language=returned_language,
                native_mode=False,
            ),
            "video_id": video_id,
            "provider": "supadata",
            "requested_language": requested_language,
            "returned_language": returned_language,
            # `False` denotes an auto-mode request. Supadata does not expose
            # whether auto was fulfilled from source captions or hosted ASR.
            "native_mode": False,
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "segments": cues,
            "duration_sec": duration_sec,
            "created_at": created_at,
        }
    )
    if artifact is None:
        raise ProviderRequestError(
            "Supadata returned malformed or non-monotonic transcript cues.",
            provider="supadata",
            operation="transcript",
        )
    if store.is_video_tombstoned(video_id):
        raise ProviderRequestError(
            "This YouTube video was blocked during retrieval.",
            provider="supadata",
            operation="transcript",
        )
    store.put_transcript(artifact)
    return artifact


def fetch_transcript_artifact(
    url: str,
    lang: str = "en",
    should_cancel: Callable[[], bool] | None = None,
    *,
    chunk_size: int | None = None,
    context: GenerationContext | None = None,
    cache_store: ProviderCacheStore | None = None,
    deadline_monotonic: float | None = None,
) -> TranscriptArtifact:
    return run_cancellable(
        lambda: _fetch_transcript_artifact_async(
            url,
            lang,
            should_cancel,
            chunk_size=chunk_size,
            context=context,
            cache_store=cache_store,
            deadline_monotonic=deadline_monotonic,
        ),
        should_cancel,
    )


def fetch_transcript(
    url: str,
    lang: str = "en",
    chunk_size: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
    *,
    context: GenerationContext | None = None,
    cache_store: ProviderCacheStore | None = None,
    deadline_monotonic: float | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper returning timestamped transcript cue dictionaries."""
    return fetch_transcript_artifact(
        url,
        lang,
        should_cancel,
        chunk_size=chunk_size,
        context=context,
        cache_store=cache_store,
        deadline_monotonic=deadline_monotonic,
    ).segments
