# backend/app/clip_engine/run.py
"""Inline adapter for practice's fast Supadata -> Gemini -> iframe path."""
from __future__ import annotations

from . import config
from . import segment_cache
from .cancellation import raise_if_cancelled
from .clipper import embed
from .singleflight import singleflight
from ...pipeline import gemini_segment
from .errors import (
    CancellationError,
    ClipError,
    ProviderError,
    TranscriptError,
    UnsupportedURLError,
)
from .metadata import extract_video_id


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
            quality_degraded=str(event.get("operation") or "").startswith("pro_fallback"),
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
    settings.setdefault("segment_fine_snap", config.SEGMENT_FINE_SNAP)
    settings.setdefault("segment_min_clip_s", config.SEGMENT_MIN_CLIP_S)
    # A Flash response with any rejected proposal is not safe to ship partially.
    settings.setdefault("segment_accept_partial_flash", False)
    settings["should_cancel"] = should_cancel

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    transcript = _transcribe(canonical_url, video_id, settings)
    raise_if_cancelled(should_cancel)
    if not (transcript.get("segments")):
        raise TranscriptError(f"Empty transcript for {video_id}")

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
                    if clips:
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
        raise ClipError(f"Gemini segmentation failed for {video_id}: {exc}") from exc

    for c in clips:
        raise_if_cancelled(should_cancel)
        c["embed_url"] = embed.embed_url(video_id, c["start"], c["end"])
    return {"video_id": video_id, "clips": clips, "transcript": transcript, "notes": notes}


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
    runtime_settings["should_cancel"] = should_cancel
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
