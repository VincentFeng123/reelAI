# backend/app/clip_engine/run.py
"""Inline adapter for practice's fast Supadata -> Gemini -> iframe path."""
from __future__ import annotations

import math

from . import config
from . import segment_cache
from .cancellation import raise_if_cancelled
from .clipper import embed
from .singleflight import singleflight
from ...pipeline import gemini_segment
from .errors import (
    CancellationError,
    ClipError,
    ProviderBudgetExceededError,
    ProviderError,
    ProviderRequestError,
    ProviderTransientError,
    TranscriptError,
    UnsupportedURLError,
)
from .metadata import extract_video_id
from .provider_runtime import GenerationContext


def is_valid_timestamped_supadata_transcript(transcript: dict) -> bool:
    """Verify the provenance markers and cue invariants used by ingestion."""
    if (
        transcript.get("source") != "supadata"
        or not str(transcript.get("artifact_key") or "").strip()
        or not isinstance(transcript.get("native_mode"), bool)
    ):
        return False
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        return False

    seen_ids: set[str] = set()
    previous_start = -1.0
    previous_end = -1.0
    for cue in segments:
        if not isinstance(cue, dict):
            return False
        cue_id = str(cue.get("cue_id") or "").strip()
        text = " ".join(str(cue.get("text") or "").split()).strip()
        try:
            start = float(cue.get("start"))
            end = float(cue.get("end"))
        except (TypeError, ValueError):
            return False
        if (
            not cue_id
            or cue_id in seen_ids
            or not text
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or start + 1e-9 < previous_start
            or end + 1e-9 < previous_end
        ):
            return False
        seen_ids.add(cue_id)
        previous_start = start
        previous_end = end
    return True


def _segment_usage_stage(operation: object) -> str:
    normalized = str(operation or "").casefold()
    if normalized == "flash_boundary_repair":
        return "repair"
    if "enrichment" in normalized:
        return "enrichment"
    if normalized in {
        "flash_boundary_selector",
        "flash_single_candidate",
        "boundary_selection",
        "pro_authoritative",
        "pro_fallback",
    }:
        return "selection"
    return normalized or "selection"


def _transcribe(url: str, video_id: str, settings: dict) -> dict:
    """Fetch a hosted timestamped transcript as {segments, words, duration, ...}."""
    from .clipper.pipeline.transcribe import transcribe_supadata  # lazy
    try:
        return transcribe_supadata(url, video_id, settings)
    except CancellationError:
        raise
    except ProviderError:
        raise
    except Exception as exc:  # normalize to engine error
        raise TranscriptError(f"Supadata transcript failed for {video_id}: {exc}") from exc


def _wire_segment_runtime(
    settings: dict,
    video_id: str,
    *,
    reserve_segmentation: bool = True,
) -> None:
    """Bridge the hosted job ledger into practice's unchanged segment router."""
    context = settings.get("generation_context") or settings.get("provider_context")
    if context is not None:
        if reserve_segmentation:
            context.reserve("segmentation")
        reserve = getattr(context, "reserve_gemini_call", None)
        if callable(reserve):
            settings.setdefault("_segment_budget_reserve", reserve)
        reconcile = getattr(context, "reconcile_gemini_call", None)
        if callable(reconcile):
            settings.setdefault("_segment_budget_reconcile", reconcile)
        fallback_gate = getattr(context, "allow_pro_fallback", None)
        if callable(fallback_gate):
            def gated_fallback(*, accepted_count: int, video_id: str = "") -> bool:
                return bool(
                    fallback_gate(
                        accepted_count=accepted_count,
                        video_id=video_id,
                        candidate_rank=settings.get("_segment_candidate_rank"),
                        deadline_monotonic=settings.get("deadline_monotonic"),
                    )
                )

            settings.setdefault("_segment_pro_fallback_gate", gated_fallback)

    existing_sink = settings.get("_segment_telemetry")

    def sink(event: dict) -> None:
        if callable(existing_sink):
            existing_sink(event)
        if context is None:
            return
        record_event = getattr(context, "record_segment_event", None)
        if callable(record_event):
            record_event(event)
        if event.get("event") != "model_call":
            return
        has_provider_response = bool(
            event.get("finish_reason")
            or event.get("prompt_tokens")
            or event.get("prompt_token_count")
            or event.get("candidate_tokens")
            or event.get("candidates_token_count")
            or event.get("thought_tokens")
            or event.get("thoughts_token_count")
            or event.get("total_tokens")
            or event.get("total_token_count")
        )
        context.record_gemini(
            stage=_segment_usage_stage(event.get("operation")),
            attempt=max(1, int(event.get("retries") or 0) + 1),
            model_used=str(event.get("model") or ""),
            quality_degraded=(
                bool(event.get("quality_degraded"))
                or str(event.get("operation") or "").startswith("pro_fallback")
            ),
            usage=event,
            status_code=200 if has_provider_response else None,
            error_code="" if has_provider_response else "model_call_failed",
        )

    settings["_segment_telemetry"] = sink
    settings["_segment_cancelled"] = settings.get("should_cancel")
    settings["video_id"] = video_id


def clip(url: str, topic: str, settings: dict | None = None, *, should_cancel=None) -> dict:
    raise_if_cancelled(should_cancel)
    video_id = extract_video_id(url)
    if not video_id:
        raise UnsupportedURLError(f"Not a recognized YouTube URL: {url}")
    settings = dict(settings or {})
    if not (settings.get("generation_context") or settings.get("provider_context")):
        # Every production-facing clip invocation needs the same pre-dispatch
        # token and cost guard. Callers with a multi-source job pass a shared
        # context; isolated/legacy callers receive a bounded local one.
        settings["generation_context"] = GenerationContext(
            "fast",
            generation_id=f"clip:{video_id}",
        )
    settings.setdefault("segment_fine_snap", config.SEGMENT_FINE_SNAP)
    # Candidates are independently gated; one weak proposal must not poison others.
    settings.setdefault("segment_accept_partial_flash", True)
    # This adapter is the production selector boundary. One normal Flash call is
    # authoritative; stale env or request settings cannot reactivate Pro/hybrid.
    settings["_segment_routing_mode"] = "flash_only"
    settings["_segment_thinking_level"] = "medium"
    settings["_segment_allow_flash_lite_failover"] = False
    settings["should_cancel"] = should_cancel

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    settings["_segment_video_url"] = canonical_url
    settings["_segment_video_grounding_required"] = False
    settings["_segment_media_resolution"] = "low"
    transcript = _transcribe(canonical_url, video_id, settings)
    raise_if_cancelled(should_cancel)
    if not (transcript.get("segments")):
        raise TranscriptError(f"Empty transcript for {video_id}")
    context = settings.get("generation_context") or settings.get("provider_context")
    trusted_transcript = is_valid_timestamped_supadata_transcript(transcript)
    transcript_usage_recorded = context is not None and trusted_transcript
    if transcript_usage_recorded:
        # Record transcript success before Gemini runs so a selector outage is
        # never misreported as missing captions.
        context.increment_counter("usable_transcripts")

    try:
        def run_segmenter() -> tuple[list[dict], str]:
            _wire_segment_runtime(settings, video_id)
            return gemini_segment.segment_clips(
                transcript,
                settings,
                topic=topic or "",
                video_id=video_id,
            )

        if not segment_cache.cache_enabled():
            clips, notes = run_segmenter()
        else:
            cache_key = segment_cache.segment_cache_key(
                video_id=video_id,
                topic=topic or "",
                transcript=transcript,
                settings=settings,
            )
            with singleflight(cache_key, should_cancel):
                cached = segment_cache.load_segment_result(
                    cache_key,
                    video_id=video_id,
                    transcript=transcript,
                    settings=settings,
                )
                if cached is not None:
                    clips, notes = cached
                    context = settings.get("generation_context") or settings.get("provider_context")
                    if context is not None:
                        context.increment_counter("segmentation_cache_hits")
                        context.record_cache_hit(
                            provider="gemini",
                            operation="segmentation",
                            metadata={"cache_key": cache_key},
                        )
                else:
                    clips, notes = run_segmenter()
                    segment_cache.store_segment_result(
                        cache_key,
                        clips,
                        notes,
                        video_id=video_id,
                        transcript=transcript,
                        settings=settings,
                    )
        raise_if_cancelled(should_cancel)
    except CancellationError:
        raise
    except ProviderError:
        raise
    except Exception as exc:
        raise_if_cancelled(should_cancel)
        telemetry = getattr(exc, "telemetry", None)
        if isinstance(telemetry, dict):
            raw_status = telemetry.get("provider_status_code")
            try:
                status_code = int(raw_status) if raw_status is not None else None
            except (TypeError, ValueError, OverflowError):
                status_code = None
            error_type = str(telemetry.get("error_type") or type(exc).__name__)
            provider_error_type = str(
                telemetry.get("provider_error_type") or ""
            )
            raw_retryable = telemetry.get("retryable")
            if error_type == "ProviderBudgetExceededError":
                raise ProviderBudgetExceededError(
                    "Clip selection budget was exhausted before dispatch.",
                    provider="gemini",
                    operation="segmentation",
                    detail=error_type,
                ) from exc
            if isinstance(raw_retryable, bool):
                transient = raw_retryable
            else:
                transient = error_type in {
                    "GeminiDeadlineExceededError",
                    "GeminiTransportError",
                } and status_code is None
            error_class = ProviderTransientError if transient else ProviderRequestError
            message = (
                "Gemini is temporarily unavailable."
                if transient
                else "Gemini rejected the segmentation request."
            )
            raise error_class(
                message,
                provider="gemini",
                operation="segmentation",
                status_code=status_code,
                detail=(
                    f"{error_type}:{provider_error_type}"
                    if provider_error_type
                    else error_type
                ),
            ) from exc
        raise ClipError(f"Gemini segmentation failed for {video_id}: {exc}") from exc

    for c in clips:
        raise_if_cancelled(should_cancel)
        c["embed_url"] = embed.embed_url(video_id, c["start"], c["end"])
    return {
        "video_id": video_id,
        "clips": clips,
        "transcript": transcript,
        "notes": notes,
        "_transcript_usage_recorded": transcript_usage_recorded,
    }


def pro_boundary_fallback(
    transcript: dict,
    *,
    topic: str,
    video_id: str,
    settings: dict | None = None,
    should_cancel=None,
) -> dict:
    """Run aggregate Pro selection without retrieving or re-timestamping transcript data."""
    raise_if_cancelled(should_cancel)
    runtime_settings = dict(settings or {})
    if not (
        runtime_settings.get("generation_context")
        or runtime_settings.get("provider_context")
    ):
        runtime_settings["generation_context"] = GenerationContext(
            "fast",
            generation_id=f"boundary:{video_id}",
        )
    runtime_settings["should_cancel"] = should_cancel
    runtime_settings["_segment_routing_mode"] = "pro_only"
    runtime_settings["_segment_video_url"] = (
        f"https://www.youtube.com/watch?v={video_id}"
    )
    runtime_settings["_segment_video_grounding_required"] = False
    runtime_settings["_segment_media_resolution"] = "low"
    _wire_segment_runtime(
        runtime_settings,
        video_id,
        reserve_segmentation=False,
    )
    result = gemini_segment.pro_boundary_fallback_detailed(
        transcript,
        runtime_settings,
        topic=topic,
        video_id=video_id,
        # This adapter is an explicit standalone selector, not the disabled
        # automatic hybrid fallback. It uses the ordinary bounded selector slot
        # while retaining the boundary-only prompt.
        budget_operation="pro_authoritative",
    )
    clips = list(result.clips)
    for clip in clips:
        clip["embed_url"] = embed.embed_url(
            video_id,
            clip["start"],
            clip["end"],
        )
    return {
        "video_id": video_id,
        "clips": clips,
        "transcript": transcript,
        "notes": result.notes,
        "route": result.route,
    }
